use std::sync::{Mutex, OnceLock};

use burn_backend::{FloatDType, IntDType, Shape, TensorMetadata};

use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use crate::{GnaBackendError, GnaHwDevice, GnaLibrary, Result, ffi};

pub mod ops;

static GNA_EXEC_LOCK: Mutex<()> = Mutex::new(());
static GNA_LIBRARY: OnceLock<Option<GnaLibrary>> = OnceLock::new();

fn get_library() -> Result<&'static GnaLibrary> {
    let lib = GNA_LIBRARY.get_or_init(|| GnaLibrary::load_default().ok());
    lib.as_ref().ok_or(GnaBackendError::DeviceNotInitialized(0))
}

const fn align_up(value: usize, align: usize) -> usize {
    value.div_ceil(align) * align
}

/// Execute `lhs [m, k] * rhs [k, n] (+ bias [n])` strictly on Intel GNA hardware.
pub fn gna_affine_i16(
    lhs: &[i16],
    rhs: &[i16],
    bias: Option<&[i32]>,
    m: usize,
    k: usize,
    n: usize,
    device_index: u32,
) -> Result<Vec<i32>> {
    if m == 0 || k == 0 || n == 0 {
        return Ok(vec![0i32; m * n]);
    }

    let _guard = GNA_EXEC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let lib = get_library()?;
    let hw_device = GnaHwDevice::open(lib, device_index)?;

    let w_pad = k.div_ceil(64).max(1) * 64;
    let h_pad = n.div_ceil(64).max(1) * 64;

    let inputs_bytes = w_pad * std::mem::size_of::<i16>();
    let outputs_bytes = h_pad * std::mem::size_of::<i32>();
    let weights_bytes = h_pad * w_pad * std::mem::size_of::<i16>();
    let biases_bytes = h_pad * std::mem::size_of::<i32>();

    let outputs_offset = align_up(inputs_bytes, 4096);
    let weights_offset = align_up(outputs_offset + outputs_bytes, 4096);
    let biases_offset = align_up(weights_offset + weights_bytes, 4096);
    let total_bytes = align_up(biases_offset + biases_bytes, 4096);

    let memory = hw_device.allocate_buffer(total_bytes)?;
    let base = memory.as_raw_ptr() as *mut u8;

    let inputs_ptr = base as *mut i16;
    let outputs_ptr = unsafe { base.add(outputs_offset) } as *mut i32;
    let weights_ptr = unsafe { base.add(weights_offset) } as *mut i16;
    let biases_ptr = unsafe { base.add(biases_offset) } as *mut i32;

    // Zero-initialize weights & biases, then transpose rhs [k, n] into weights [h_pad, w_pad].
    unsafe {
        std::ptr::write_bytes(weights_ptr, 0, h_pad * w_pad);
        std::ptr::write_bytes(biases_ptr, 0, h_pad);
        for row in 0..k {
            for col in 0..n {
                *weights_ptr.add(col * w_pad + row) = rhs[row * n + col];
            }
        }
        if let Some(b) = bias {
            std::ptr::copy_nonoverlapping(b.as_ptr(), biases_ptr, n.min(b.len()));
        }
    }

    let model = ffi::GnaModelBuilder::new()
        .add_fully_connected_affine(
            ffi::types::Gna2Tensor::d2(
                w_pad as u32,
                1,
                ffi::types::Gna2DataType::Int16,
                inputs_ptr as _,
            ),
            ffi::types::Gna2Tensor::d2(
                h_pad as u32,
                1,
                ffi::types::Gna2DataType::Int32,
                outputs_ptr as _,
            ),
            ffi::types::Gna2Tensor::d2(
                h_pad as u32,
                w_pad as u32,
                ffi::types::Gna2DataType::Int16,
                weights_ptr as _,
            ),
            ffi::types::Gna2Tensor::d1(
                h_pad as u32,
                ffi::types::Gna2DataType::Int32,
                biases_ptr as _,
            ),
            None,
        )
        .build(&hw_device)?;

    let mut request = ffi::GnaRequestConfig::create(hw_device.library(), model.id())?;
    unsafe {
        request.set_operand_buffer(0, 0, inputs_ptr as _)?;
        request.set_operand_buffer(0, 1, outputs_ptr as _)?;
    }
    request.set_acceleration_mode(ffi::types::Gna2AccelerationMode::Hardware)?;

    let mut result = vec![0i32; m * n];
    for r in 0..m {
        unsafe {
            std::ptr::write_bytes(inputs_ptr, 0, w_pad);
            std::ptr::copy_nonoverlapping(lhs[r * k..].as_ptr(), inputs_ptr, k);
            std::ptr::write_bytes(outputs_ptr, 0, h_pad);
        }
        let req_id = request.enqueue()?;
        request.wait(req_id, 1000)?;
        unsafe {
            std::ptr::copy_nonoverlapping(outputs_ptr, result[r * n..].as_mut_ptr(), n);
        }
    }

    drop(request);
    drop(model);
    drop(memory);
    drop(hw_device);

    Ok(result)
}

/// Execute `lhs [m, k] * rhs [k, n]` strictly on Intel GNA hardware.
pub fn gna_matmul_i16(
    lhs: &[i16],
    rhs: &[i16],
    m: usize,
    k: usize,
    n: usize,
    device_index: u32,
) -> Result<Vec<i32>> {
    gna_affine_i16(lhs, rhs, None, m, k, n, device_index)
}

pub(crate) fn decode_ints(bytes: &[u8], dtype: IntDType) -> Vec<i64> {
    match dtype {
        IntDType::I8 => bytes.iter().map(|&b| (b as i8) as i64).collect(),
        IntDType::U8 => bytes.iter().map(|&b| b as i64).collect(),
        IntDType::I16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| i16::from_le_bytes(c) as i64)
            .collect(),
        IntDType::U16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| u16::from_le_bytes(c) as i64)
            .collect(),
        IntDType::I32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| i32::from_le_bytes(c) as i64)
            .collect(),
        IntDType::U32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c) as i64)
            .collect(),
        IntDType::I64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| i64::from_le_bytes(c))
            .collect(),
        IntDType::U64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| u64::from_le_bytes(c) as i64)
            .collect(),
    }
}

pub(crate) fn encode_ints(values: &[i64], dtype: IntDType) -> Vec<u8> {
    match dtype {
        IntDType::I8 => values.iter().map(|&v| (v as i8) as u8).collect(),
        IntDType::U8 => values.iter().map(|&v| v as u8).collect(),
        IntDType::I16 => values
            .iter()
            .flat_map(|&v| (v as i16).to_le_bytes())
            .collect(),
        IntDType::U16 => values
            .iter()
            .flat_map(|&v| (v as u16).to_le_bytes())
            .collect(),
        IntDType::I32 => values
            .iter()
            .flat_map(|&v| (v as i32).to_le_bytes())
            .collect(),
        IntDType::U32 => values
            .iter()
            .flat_map(|&v| (v as u32).to_le_bytes())
            .collect(),
        IntDType::I64 => values.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        IntDType::U64 => values
            .iter()
            .flat_map(|&v| (v as u64).to_le_bytes())
            .collect(),
    }
}

pub(crate) fn decode_floats(bytes: &[u8], dtype: FloatDType) -> Vec<f32> {
    match dtype {
        FloatDType::F64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| f64::from_le_bytes(c) as f32)
            .collect(),
        _ => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect(),
    }
}

pub(crate) fn encode_floats(values: &[f32], dtype: FloatDType) -> Vec<u8> {
    match dtype {
        FloatDType::F64 => values
            .iter()
            .flat_map(|&v| (v as f64).to_le_bytes())
            .collect(),
        _ => values.iter().flat_map(|&v| v.to_le_bytes()).collect(),
    }
}

/// Execute N-D integer matrix multiplication strictly on GNA hardware.
pub fn execute_int_matmul(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let lhs_dims: &[usize] = &lhs_shape;
    let rhs_dims: &[usize] = &rhs_shape;

    if lhs_dims.len() < 2 {
        panic!("{}", GnaBackendError::InvalidMatmulRank(lhs_dims.len()));
    }
    if rhs_dims.len() < 2 {
        panic!("{}", GnaBackendError::InvalidMatmulRank(rhs_dims.len()));
    }

    let m = lhs_dims[lhs_dims.len() - 2];
    let k = lhs_dims[lhs_dims.len() - 1];
    let n = rhs_dims[rhs_dims.len() - 1];

    let mut out_dims = lhs_dims[..lhs_dims.len() - 2].to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);
    let out_dtype = lhs.dtype();

    let lhs_vals = decode_ints(lhs.buffer().as_slice(), lhs.dtype());
    let rhs_vals = decode_ints(rhs.buffer().as_slice(), rhs.dtype());

    let batch_count: usize = lhs_dims[..lhs_dims.len() - 2].iter().product();
    let lhs_stride = m * k;
    let rhs_stride = k * n;

    let mut out_vals = Vec::with_capacity(batch_count.max(1) * m * n);
    for b in 0..batch_count.max(1) {
        let l_slice = &lhs_vals[b * lhs_stride..(b + 1) * lhs_stride];
        let r_slice = &rhs_vals[b * rhs_stride..(b + 1) * rhs_stride];

        let l_i16: Vec<i16> = l_slice
            .iter()
            .map(|&v| i16::try_from(v).map_err(|_| GnaBackendError::IntValueOutOfI16Range(v)))
            .collect::<Result<Vec<_>>>()
            .expect("GNA hardware int_matmul input out of Int16 range");
        let r_i16: Vec<i16> = r_slice
            .iter()
            .map(|&v| i16::try_from(v).map_err(|_| GnaBackendError::IntValueOutOfI16Range(v)))
            .collect::<Result<Vec<_>>>()
            .expect("GNA hardware int_matmul weight out of Int16 range");

        let gna_out = gna_matmul_i16(&l_i16, &r_i16, m, k, n, 0)
            .expect("GNA hardware int_matmul execution failed");
        out_vals.extend(gna_out.into_iter().map(|v| v as i64));
    }

    let out_bytes = encode_ints(&out_vals, out_dtype);
    GnaIntTensorPrimitive::new(out_shape, out_dtype, GnaBuffer::from_bytes(out_bytes))
}

/// Execute N-D float matrix multiplication strictly on GNA hardware (via dynamic symmetric i16 quantization).
pub fn execute_float_matmul(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let lhs_dims: &[usize] = &lhs_shape;
    let rhs_dims: &[usize] = &rhs_shape;

    if lhs_dims.len() < 2 {
        panic!("{}", GnaBackendError::InvalidMatmulRank(lhs_dims.len()));
    }
    if rhs_dims.len() < 2 {
        panic!("{}", GnaBackendError::InvalidMatmulRank(rhs_dims.len()));
    }

    let m = lhs_dims[lhs_dims.len() - 2];
    let k = lhs_dims[lhs_dims.len() - 1];
    let n = rhs_dims[rhs_dims.len() - 1];

    let mut out_dims = lhs_dims[..lhs_dims.len() - 2].to_vec();
    out_dims.push(m);
    out_dims.push(n);
    let out_shape = Shape::from(out_dims);
    let out_dtype = lhs.dtype();

    let lhs_vals = decode_floats(lhs.buffer().as_slice(), lhs.dtype());
    let rhs_vals = decode_floats(rhs.buffer().as_slice(), rhs.dtype());

    let batch_count: usize = lhs_dims[..lhs_dims.len() - 2].iter().product();
    let lhs_stride = m * k;
    let rhs_stride = k * n;

    let q_max = ((i32::MAX as f64 / (k.max(1) as f64))
        .sqrt()
        .min(16384.0)
        .floor()) as f32;

    let mut out_vals = Vec::with_capacity(batch_count.max(1) * m * n);
    for b in 0..batch_count.max(1) {
        let l_slice = &lhs_vals[b * lhs_stride..(b + 1) * lhs_stride];
        let r_slice = &rhs_vals[b * rhs_stride..(b + 1) * rhs_stride];

        let max_l = l_slice.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let max_r = r_slice.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));

        let scale_l = if max_l == 0.0 { 1.0 } else { max_l / q_max };
        let scale_r = if max_r == 0.0 { 1.0 } else { max_r / q_max };

        let l_i16: Vec<i16> = l_slice
            .iter()
            .map(|&v| (v / scale_l).round().clamp(-q_max, q_max) as i16)
            .collect();
        let r_i16: Vec<i16> = r_slice
            .iter()
            .map(|&v| (v / scale_r).round().clamp(-q_max, q_max) as i16)
            .collect();

        let gna_out = gna_matmul_i16(&l_i16, &r_i16, m, k, n, 0)
            .expect("GNA hardware float_matmul execution failed");
        let combined_scale = scale_l * scale_r;
        out_vals.extend(gna_out.into_iter().map(|v| v as f32 * combined_scale));
    }

    let out_bytes = encode_floats(&out_vals, out_dtype);
    GnaFloatTensorPrimitive::new(out_shape, out_dtype, GnaBuffer::from_bytes(out_bytes))
}
