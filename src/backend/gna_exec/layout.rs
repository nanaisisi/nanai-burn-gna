use burn_backend::{Scalar, Shape, Slice, TensorMetadata};

use super::{decode_floats, decode_ints, encode_floats, encode_ints};
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;

fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

fn linear_to_coords(mut idx: usize, dims: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; dims.len()];
    for i in (0..dims.len()).rev() {
        let d = dims[i].max(1);
        coords[i] = idx % d;
        idx /= d;
    }
    coords
}

fn coords_to_linear(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides.iter())
        .map(|(&c, &s)| c * s)
        .sum()
}

pub fn permute_bytes(
    bytes: &[u8],
    dims: &[usize],
    axes: &[usize],
    elem_size: usize,
) -> (Shape, Vec<u8>) {
    let out_dims: Vec<usize> = axes.iter().map(|&a| dims[a]).collect();
    let total: usize = out_dims.iter().product();
    let in_strides = compute_strides(dims);
    let mut out = vec![0u8; total * elem_size];

    for out_idx in 0..total {
        let out_coords = linear_to_coords(out_idx, &out_dims);
        let mut in_coords = vec![0; dims.len()];
        for (o_dim, &axis) in axes.iter().enumerate() {
            in_coords[axis] = out_coords[o_dim];
        }
        let in_idx = coords_to_linear(&in_coords, &in_strides);
        out[out_idx * elem_size..(out_idx + 1) * elem_size]
            .copy_from_slice(&bytes[in_idx * elem_size..(in_idx + 1) * elem_size]);
    }

    (Shape::from(out_dims), out)
}

pub fn swap_dims_bytes(
    bytes: &[u8],
    dims: &[usize],
    dim1: usize,
    dim2: usize,
    elem_size: usize,
) -> (Shape, Vec<u8>) {
    let mut axes: Vec<usize> = (0..dims.len()).collect();
    axes.swap(dim1, dim2);
    permute_bytes(bytes, dims, &axes, elem_size)
}

pub fn flip_bytes(bytes: &[u8], dims: &[usize], axes: &[usize], elem_size: usize) -> Vec<u8> {
    let total: usize = dims.iter().product();
    let strides = compute_strides(dims);
    let mut out = vec![0u8; total * elem_size];

    for out_idx in 0..total {
        let mut coords = linear_to_coords(out_idx, dims);
        for &axis in axes {
            if axis < dims.len() {
                coords[axis] = dims[axis] - 1 - coords[axis];
            }
        }
        let in_idx = coords_to_linear(&coords, &strides);
        out[out_idx * elem_size..(out_idx + 1) * elem_size]
            .copy_from_slice(&bytes[in_idx * elem_size..(in_idx + 1) * elem_size]);
    }
    out
}

pub fn expand_bytes(
    bytes: &[u8],
    in_dims: &[usize],
    out_shape: &Shape,
    elem_size: usize,
) -> Vec<u8> {
    let out_dims: &[usize] = out_shape;
    let total: usize = out_dims.iter().product();
    let in_strides = compute_strides(in_dims);
    let rank_diff = out_dims.len().saturating_sub(in_dims.len());
    let mut out = vec![0u8; total * elem_size];

    for out_idx in 0..total {
        let out_coords = linear_to_coords(out_idx, out_dims);
        let mut in_coords = vec![0; in_dims.len()];
        for (i, &in_d) in in_dims.iter().enumerate() {
            in_coords[i] = if in_d == 1 {
                0
            } else {
                out_coords[i + rank_diff]
            };
        }
        let in_idx = coords_to_linear(&in_coords, &in_strides);
        out[out_idx * elem_size..(out_idx + 1) * elem_size]
            .copy_from_slice(&bytes[in_idx * elem_size..(in_idx + 1) * elem_size]);
    }
    out
}

fn slice_indices_for_dim(slice: &Slice, dim_size: usize) -> Vec<usize> {
    let range = slice.to_range(dim_size);
    let step = slice.step();
    if step > 0 {
        (range.start..range.end).step_by(step as usize).collect()
    } else {
        (range.start..range.end)
            .rev()
            .step_by((-step) as usize)
            .collect()
    }
}

pub fn slice_bytes(
    bytes: &[u8],
    dims: &[usize],
    slices: &[Slice],
    elem_size: usize,
) -> (Shape, Vec<u8>) {
    let dim_indices: Vec<Vec<usize>> = dims
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let s = slices.get(i).copied().unwrap_or_else(Slice::full);
            slice_indices_for_dim(&s, d)
        })
        .collect();
    let out_dims: Vec<usize> = dim_indices.iter().map(|v| v.len()).collect();
    let total: usize = out_dims.iter().product();
    let in_strides = compute_strides(dims);
    let mut out = vec![0u8; total * elem_size];

    for out_idx in 0..total {
        let out_coords = linear_to_coords(out_idx, &out_dims);
        let in_coords: Vec<usize> = out_coords
            .iter()
            .enumerate()
            .map(|(d, &c)| dim_indices[d][c])
            .collect();
        let in_idx = coords_to_linear(&in_coords, &in_strides);
        out[out_idx * elem_size..(out_idx + 1) * elem_size]
            .copy_from_slice(&bytes[in_idx * elem_size..(in_idx + 1) * elem_size]);
    }
    (Shape::from(out_dims), out)
}

pub fn slice_assign_bytes(
    bytes: &[u8],
    dims: &[usize],
    slices: &[Slice],
    val_bytes: &[u8],
    elem_size: usize,
) -> Vec<u8> {
    let dim_indices: Vec<Vec<usize>> = dims
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let s = slices.get(i).copied().unwrap_or_else(Slice::full);
            slice_indices_for_dim(&s, d)
        })
        .collect();
    let val_dims: Vec<usize> = dim_indices.iter().map(|v| v.len()).collect();
    let val_total: usize = val_dims.iter().product();
    let out_strides = compute_strides(dims);
    let mut out = bytes.to_vec();

    for val_idx in 0..val_total {
        let val_coords = linear_to_coords(val_idx, &val_dims);
        let dst_coords: Vec<usize> = val_coords
            .iter()
            .enumerate()
            .map(|(d, &c)| dim_indices[d][c])
            .collect();
        let dst_idx = coords_to_linear(&dst_coords, &out_strides);
        out[dst_idx * elem_size..(dst_idx + 1) * elem_size]
            .copy_from_slice(&val_bytes[val_idx * elem_size..(val_idx + 1) * elem_size]);
    }
    out
}

pub fn select_bytes(
    bytes: &[u8],
    dims: &[usize],
    dim: usize,
    indices: &[i64],
    elem_size: usize,
) -> (Shape, Vec<u8>) {
    let mut out_dims = dims.to_vec();
    out_dims[dim] = indices.len();
    let total: usize = out_dims.iter().product();
    let in_strides = compute_strides(dims);
    let mut out = vec![0u8; total * elem_size];

    for out_idx in 0..total {
        let mut coords = linear_to_coords(out_idx, &out_dims);
        coords[dim] = indices[coords[dim]] as usize;
        let in_idx = coords_to_linear(&coords, &in_strides);
        out[out_idx * elem_size..(out_idx + 1) * elem_size]
            .copy_from_slice(&bytes[in_idx * elem_size..(in_idx + 1) * elem_size]);
    }
    (Shape::from(out_dims), out)
}

pub fn gather_bytes(
    dim: usize,
    bytes: &[u8],
    dims: &[usize],
    idx_vals: &[i64],
    idx_dims: &[usize],
    elem_size: usize,
) -> (Shape, Vec<u8>) {
    let total: usize = idx_dims.iter().product();
    let in_strides = compute_strides(dims);
    let mut out = vec![0u8; total * elem_size];

    for out_idx in 0..total {
        let mut coords = linear_to_coords(out_idx, idx_dims);
        coords[dim] = idx_vals[out_idx] as usize;
        let in_idx = coords_to_linear(&coords, &in_strides);
        out[out_idx * elem_size..(out_idx + 1) * elem_size]
            .copy_from_slice(&bytes[in_idx * elem_size..(in_idx + 1) * elem_size]);
    }
    (Shape::from(idx_dims.to_vec()), out)
}

pub fn mask_where_bytes(
    bytes: &[u8],
    mask_bytes: &[u8],
    val_bytes: &[u8],
    elem_size: usize,
) -> Vec<u8> {
    let total = mask_bytes.len();
    let mut out = bytes.to_vec();
    for i in 0..total {
        if mask_bytes[i] != 0 {
            let v_off = (i * elem_size) % val_bytes.len();
            out[i * elem_size..(i + 1) * elem_size]
                .copy_from_slice(&val_bytes[v_off..v_off + elem_size]);
        }
    }
    out
}

pub fn mask_fill_bytes(
    bytes: &[u8],
    mask_bytes: &[u8],
    fill_elem: &[u8],
    elem_size: usize,
) -> Vec<u8> {
    let total = mask_bytes.len();
    let mut out = bytes.to_vec();
    for i in 0..total {
        if mask_bytes[i] != 0 {
            out[i * elem_size..(i + 1) * elem_size].copy_from_slice(fill_elem);
        }
    }
    out
}

pub fn scalar_to_le_bytes(scalar: Scalar, elem_size: usize) -> Vec<u8> {
    match elem_size {
        1 => vec![scalar.elem::<u8>()],
        2 => scalar.elem::<i16>().to_le_bytes().to_vec(),
        4 => scalar.elem::<f32>().to_le_bytes().to_vec(),
        8 => scalar.elem::<f64>().to_le_bytes().to_vec(),
        _ => vec![0u8; elem_size],
    }
}

pub fn int_scalar_to_le_bytes(scalar: Scalar, elem_size: usize) -> Vec<u8> {
    match elem_size {
        1 => vec![scalar.elem::<i8>() as u8],
        2 => scalar.elem::<i16>().to_le_bytes().to_vec(),
        4 => scalar.elem::<i32>().to_le_bytes().to_vec(),
        8 => scalar.elem::<i64>().to_le_bytes().to_vec(),
        _ => vec![0u8; elem_size],
    }
}

#[derive(Clone, Copy)]
pub enum CmpOp {
    Equal,
    Greater,
    GreaterEqual,
    Lower,
    LowerEqual,
}

fn eval_cmp_f32(diff: &[f32], op: CmpOp) -> Vec<u8> {
    const EPS: f32 = 1e-3;
    diff.iter()
        .map(|&d| {
            let b = match op {
                CmpOp::Equal => d.abs() <= EPS,
                CmpOp::Greater => d > EPS,
                CmpOp::GreaterEqual => d >= -EPS,
                CmpOp::Lower => d < -EPS,
                CmpOp::LowerEqual => d <= EPS,
            };
            u8::from(b)
        })
        .collect()
}

fn eval_cmp_i64(diff: &[i64], op: CmpOp) -> Vec<u8> {
    diff.iter()
        .map(|&d| {
            let b = match op {
                CmpOp::Equal => d == 0,
                CmpOp::Greater => d > 0,
                CmpOp::GreaterEqual => d >= 0,
                CmpOp::Lower => d < 0,
                CmpOp::LowerEqual => d <= 0,
            };
            u8::from(b)
        })
        .collect()
}

pub fn compare_float(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
    op: CmpOp,
) -> GnaBoolTensorPrimitive {
    let diff = super::ops::execute_float_add_sub(lhs, rhs, -1);
    let shape = diff.shape();
    let vals = decode_floats(diff.buffer().as_slice(), diff.dtype());
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(eval_cmp_f32(&vals, op)))
}

pub fn compare_float_elem(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
    op: CmpOp,
) -> GnaBoolTensorPrimitive {
    let diff = super::ops::execute_float_add_scalar(lhs, rhs, -1.0);
    let shape = diff.shape();
    let vals = decode_floats(diff.buffer().as_slice(), diff.dtype());
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(eval_cmp_f32(&vals, op)))
}

pub fn compare_int(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
    op: CmpOp,
) -> GnaBoolTensorPrimitive {
    let diff = super::ops::execute_int_add_sub(lhs, rhs, -1);
    let shape = diff.shape();
    let vals = decode_ints(diff.buffer().as_slice(), diff.dtype());
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(eval_cmp_i64(&vals, op)))
}

pub fn compare_int_elem(
    lhs: GnaIntTensorPrimitive,
    rhs: Scalar,
    op: CmpOp,
) -> GnaBoolTensorPrimitive {
    let diff = super::ops::execute_int_add_scalar(lhs, rhs, -1);
    let shape = diff.shape();
    let vals = decode_ints(diff.buffer().as_slice(), diff.dtype());
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(eval_cmp_i64(&vals, op)))
}

pub fn convert_float_to_int(
    tensor: GnaFloatTensorPrimitive,
    out_dtype: burn_backend::IntDType,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let vals: Vec<i64> = decode_floats(tensor.buffer().as_slice(), tensor.dtype())
        .into_iter()
        .map(|v| v as i64)
        .collect();
    GnaIntTensorPrimitive::new(
        shape,
        out_dtype,
        GnaBuffer::from_bytes(encode_ints(&vals, out_dtype)),
    )
}

pub fn convert_int_to_float(
    tensor: GnaIntTensorPrimitive,
    out_dtype: burn_backend::FloatDType,
) -> GnaFloatTensorPrimitive {
    let shape = tensor.shape();
    let vals: Vec<f32> = decode_ints(tensor.buffer().as_slice(), tensor.dtype())
        .into_iter()
        .map(|v| v as f32)
        .collect();
    GnaFloatTensorPrimitive::new(
        shape,
        out_dtype,
        GnaBuffer::from_bytes(encode_floats(&vals, out_dtype)),
    )
}

pub fn unfold_bytes(
    bytes: &[u8],
    dims: &[usize],
    params: crate::backend::UnfoldParams,
    elem_size: usize,
) -> (Shape, Vec<u8>) {
    let windows = (dims[params.dim].saturating_sub(params.size)) / params.step + 1;
    let mut out_dims = dims.to_vec();
    out_dims[params.dim] = windows;
    out_dims.push(params.size);

    let total: usize = out_dims.iter().product();
    let in_strides = compute_strides(dims);
    let mut out = vec![0u8; total * elem_size];

    for out_idx in 0..total {
        let out_coords = linear_to_coords(out_idx, &out_dims);
        let mut in_coords = out_coords[..dims.len()].to_vec();
        in_coords[params.dim] = out_coords[params.dim] * params.step + out_coords[dims.len()];
        let in_idx = coords_to_linear(&in_coords, &in_strides);
        out[out_idx * elem_size..(out_idx + 1) * elem_size]
            .copy_from_slice(&bytes[in_idx * elem_size..(in_idx + 1) * elem_size]);
    }
    (Shape::from(out_dims), out)
}
