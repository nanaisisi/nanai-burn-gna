use burn_backend::{Shape, TensorMetadata};

use crate::backend::gna_exec::codec::{decode_floats, decode_ints, encode_floats, encode_ints};
use crate::backend::gna_exec::{GnaMatmulDims, gna_matmul_i16};
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use crate::{GnaBackendError, Result};

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

        let gna_out = gna_matmul_i16(&l_i16, &r_i16, GnaMatmulDims::new(m, k, n, 0))
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

        let gna_out = gna_matmul_i16(&l_i16, &r_i16, GnaMatmulDims::new(m, k, n, 0))
            .expect("GNA hardware float_matmul execution failed");
        let combined_scale = scale_l * scale_r;
        out_vals.extend(gna_out.into_iter().map(|v| v as f32 * combined_scale));
    }

    let out_bytes = encode_floats(&out_vals, out_dtype);
    GnaFloatTensorPrimitive::new(out_shape, out_dtype, GnaBuffer::from_bytes(out_bytes))
}
