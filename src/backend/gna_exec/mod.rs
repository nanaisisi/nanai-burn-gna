use std::sync::{Mutex, OnceLock};

use crate::{GnaBackendError, GnaHwDevice, GnaLibrary, Result, ffi};

pub mod codec;
pub mod extra;
pub mod flash_attn;
pub mod layout;
pub mod matmul;
pub mod module_ops;
pub mod ops;

pub(crate) use codec::{decode_floats, decode_ints, encode_floats, encode_ints};
pub use matmul::{execute_float_matmul, execute_int_matmul};

static GNA_EXEC_LOCK: Mutex<()> = Mutex::new(());
static GNA_LIBRARY: OnceLock<Option<GnaLibrary>> = OnceLock::new();
static LAST_PERF_STATS: Mutex<Option<ffi::GnaPerformanceStats>> = Mutex::new(None);

pub fn last_performance_stats() -> Option<ffi::GnaPerformanceStats> {
    LAST_PERF_STATS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

fn get_library() -> Result<&'static GnaLibrary> {
    let lib = GNA_LIBRARY.get_or_init(|| GnaLibrary::load_default().ok());
    lib.as_ref().ok_or(GnaBackendError::DeviceNotInitialized(0))
}

const fn align_up(value: usize, align: usize) -> usize {
    value.div_ceil(align) * align
}

/// Matrix dimensions and target device index for GNA hardware affine/matmul execution.
#[derive(Clone, Copy, Debug)]
pub struct GnaMatmulDims {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub device_index: u32,
}

impl GnaMatmulDims {
    pub const fn new(m: usize, k: usize, n: usize, device_index: u32) -> Self {
        Self {
            m,
            k,
            n,
            device_index,
        }
    }
}

/// Operand slices for GNA hardware affine/matmul execution.
#[derive(Clone, Copy)]
pub struct GnaAffineOperands<'a> {
    pub lhs: &'a [i16],
    pub rhs: &'a [i16],
    pub bias: Option<&'a [i32]>,
}

/// Execute `lhs [m, k] * rhs [k, n] (+ bias [n])` strictly on Intel GNA hardware.
pub fn gna_affine_i16(ops: GnaAffineOperands<'_>, dims: GnaMatmulDims) -> Result<Vec<i32>> {
    let GnaAffineOperands { lhs, rhs, bias } = ops;
    let GnaMatmulDims {
        m,
        k,
        n,
        device_index,
    } = dims;
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

    let mut memory = hw_device.allocate_buffer(total_bytes)?;
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
    let _ = request.enable_performance_counter();

    let mut result = vec![0i32; m * n];
    let mut row_buf = vec![0i16; w_pad];
    for r in 0..m {
        row_buf.fill(0);
        row_buf[..k].copy_from_slice(&lhs[r * k..(r + 1) * k]);
        memory.copy_from_typed_slice(&row_buf)?;
        unsafe {
            std::ptr::write_bytes(outputs_ptr, 0, h_pad);
        }
        let req_id = request.enqueue()?;
        request.wait(req_id, 1000)?;
        unsafe {
            std::ptr::copy_nonoverlapping(outputs_ptr, result[r * n..].as_mut_ptr(), n);
        }
    }

    if let Ok(stats) = request.get_performance_stats() {
        *LAST_PERF_STATS.lock().unwrap_or_else(|e| e.into_inner()) = Some(stats);
    }

    drop(request);
    drop(model);
    drop(memory);
    drop(hw_device);

    Ok(result)
}

/// Execute `lhs [m, k] * rhs [k, n]` strictly on Intel GNA hardware.
pub fn gna_matmul_i16(lhs: &[i16], rhs: &[i16], dims: GnaMatmulDims) -> Result<Vec<i32>> {
    gna_affine_i16(
        GnaAffineOperands {
            lhs,
            rhs,
            bias: None,
        },
        dims,
    )
}
