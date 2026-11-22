use burn_backend::{IntDType, Shape, TensorMetadata};

use crate::backend::gna_exec::{decode_floats, decode_ints, encode_ints};
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;

#[derive(Clone, Copy)]
pub enum CumMode {
    Sum,
    Prod,
    Min,
    Max,
}

fn reduce_dim_shape(dims: &[usize], dim: usize) -> (Shape, usize, usize, usize) {
    let mut out_dims = dims.to_vec();
    out_dims[dim] = 1;
    let outer: usize = dims[..dim].iter().product();
    let dim_len = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    (Shape::from(out_dims), outer, dim_len, inner)
}

pub fn execute_int_argmax_argmin(
    tensor: GnaIntTensorPrimitive,
    dim: usize,
    is_max: bool,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let vals = decode_ints(tensor.buffer().as_slice(), tensor.dtype());
    let (out_shape, outer, dim_len, inner) = reduce_dim_shape(dims, dim);
    let mut out = Vec::with_capacity(outer * inner);

    for o in 0..outer {
        for i in 0..inner {
            let mut best_idx = 0usize;
            let mut best_val = vals[o * dim_len * inner + i];
            for d in 1..dim_len {
                let v = vals[(o * dim_len + d) * inner + i];
                if (is_max && v > best_val) || (!is_max && v < best_val) {
                    best_val = v;
                    best_idx = d;
                }
            }
            out.push(best_idx as i64);
        }
    }
    GnaIntTensorPrimitive::new(
        out_shape,
        IntDType::I32,
        GnaBuffer::from_bytes(encode_ints(&out, IntDType::I32)),
    )
}

pub fn execute_float_argmax_argmin(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
    out_dtype: IntDType,
    is_max: bool,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let vals = decode_floats(tensor.buffer().as_slice(), tensor.dtype());
    let (out_shape, outer, dim_len, inner) = reduce_dim_shape(dims, dim);
    let mut out = Vec::with_capacity(outer * inner);

    for o in 0..outer {
        for i in 0..inner {
            let mut best_idx = 0usize;
            let mut best_val = vals[o * dim_len * inner + i];
            for d in 1..dim_len {
                let v = vals[(o * dim_len + d) * inner + i];
                if (is_max && v > best_val) || (!is_max && v < best_val) {
                    best_val = v;
                    best_idx = d;
                }
            }
            out.push(best_idx as i64);
        }
    }
    GnaIntTensorPrimitive::new(
        out_shape,
        out_dtype,
        GnaBuffer::from_bytes(encode_ints(&out, out_dtype)),
    )
}

pub fn execute_int_argtopk(
    tensor: GnaIntTensorPrimitive,
    dim: usize,
    k: usize,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let vals = decode_ints(tensor.buffer().as_slice(), tensor.dtype());
    let mut out_dims = dims.to_vec();
    out_dims[dim] = k;
    let outer: usize = dims[..dim].iter().product();
    let dim_len = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    let mut out = vec![0i64; outer * k * inner];

    for o in 0..outer {
        for i in 0..inner {
            let mut pairs: Vec<(usize, i64)> = (0..dim_len)
                .map(|d| (d, vals[(o * dim_len + d) * inner + i]))
                .collect();
            pairs.sort_by_key(|a| std::cmp::Reverse(a.1));
            for (rank, (idx, _)) in pairs.into_iter().take(k).enumerate() {
                out[(o * k + rank) * inner + i] = idx as i64;
            }
        }
    }
    GnaIntTensorPrimitive::new(
        Shape::from(out_dims),
        IntDType::I32,
        GnaBuffer::from_bytes(encode_ints(&out, IntDType::I32)),
    )
}

pub fn execute_float_argtopk(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
    k: usize,
    out_dtype: IntDType,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let vals = decode_floats(tensor.buffer().as_slice(), tensor.dtype());
    let mut out_dims = dims.to_vec();
    out_dims[dim] = k;
    let outer: usize = dims[..dim].iter().product();
    let dim_len = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    let mut out = vec![0i64; outer * k * inner];

    for o in 0..outer {
        for i in 0..inner {
            let mut pairs: Vec<(usize, f32)> = (0..dim_len)
                .map(|d| (d, vals[(o * dim_len + d) * inner + i]))
                .collect();
            pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (rank, (idx, _)) in pairs.into_iter().take(k).enumerate() {
                out[(o * k + rank) * inner + i] = idx as i64;
            }
        }
    }
    GnaIntTensorPrimitive::new(
        Shape::from(out_dims),
        out_dtype,
        GnaBuffer::from_bytes(encode_ints(&out, out_dtype)),
    )
}

pub fn execute_int_cum(
    tensor: GnaIntTensorPrimitive,
    dim: usize,
    mode: CumMode,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let dtype = tensor.dtype();
    let vals = decode_ints(tensor.buffer().as_slice(), dtype);
    let outer: usize = dims[..dim].iter().product();
    let dim_len = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    let mut out = vec![0i64; vals.len()];

    for o in 0..outer {
        for i in 0..inner {
            let mut acc = vals[o * dim_len * inner + i];
            out[o * dim_len * inner + i] = acc;
            for d in 1..dim_len {
                let idx = (o * dim_len + d) * inner + i;
                let v = vals[idx];
                acc = match mode {
                    CumMode::Sum => acc.wrapping_add(v),
                    CumMode::Prod => acc.wrapping_mul(v),
                    CumMode::Min => acc.min(v),
                    CumMode::Max => acc.max(v),
                };
                out[idx] = acc;
            }
        }
    }
    GnaIntTensorPrimitive::new(shape, dtype, GnaBuffer::from_bytes(encode_ints(&out, dtype)))
}

pub fn execute_float_cum(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
    mode: CumMode,
) -> GnaFloatTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let dtype = tensor.dtype();
    let vals = decode_floats(tensor.buffer().as_slice(), dtype);
    let outer: usize = dims[..dim].iter().product();
    let dim_len = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    let mut out = vec![0.0f32; vals.len()];

    for o in 0..outer {
        for i in 0..inner {
            let mut acc = vals[o * dim_len * inner + i];
            out[o * dim_len * inner + i] = acc;
            for d in 1..dim_len {
                let idx = (o * dim_len + d) * inner + i;
                let v = vals[idx];
                acc = match mode {
                    CumMode::Sum => acc + v,
                    CumMode::Prod => acc * v,
                    CumMode::Min => acc.min(v),
                    CumMode::Max => acc.max(v),
                };
                out[idx] = acc;
            }
        }
    }
    GnaFloatTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(crate::backend::gna_exec::encode_floats(&out, dtype)),
    )
}

pub fn execute_int_prod(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
    let dtype = tensor.dtype();
    let prod: i64 = decode_ints(tensor.buffer().as_slice(), dtype)
        .into_iter()
        .product();
    GnaIntTensorPrimitive::new(
        Shape::from([1]),
        dtype,
        GnaBuffer::from_bytes(encode_ints(&[prod], dtype)),
    )
}

pub fn execute_int_prod_dim(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let dtype = tensor.dtype();
    let vals = decode_ints(tensor.buffer().as_slice(), dtype);
    let (out_shape, outer, dim_len, inner) = reduce_dim_shape(dims, dim);
    let mut out = Vec::with_capacity(outer * inner);

    for o in 0..outer {
        for i in 0..inner {
            let p: i64 = (0..dim_len)
                .map(|d| vals[(o * dim_len + d) * inner + i])
                .product();
            out.push(p);
        }
    }
    GnaIntTensorPrimitive::new(
        out_shape,
        dtype,
        GnaBuffer::from_bytes(encode_ints(&out, dtype)),
    )
}
