use burn_backend::{Shape, Slice, TensorMetadata};

use super::shape::{ByteTensorView, compute_strides, coords_to_linear, linear_to_coords};
use crate::backend::gna_exec::{decode_floats, decode_ints, encode_floats, encode_ints};
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;

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

pub struct MaskedBytes<'a> {
    bytes: &'a [u8],
    mask: &'a [u8],
}

impl<'a> MaskedBytes<'a> {
    pub const fn new(bytes: &'a [u8], mask: &'a [u8]) -> Self {
        Self { bytes, mask }
    }

    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub const fn mask(&self) -> &'a [u8] {
        self.mask
    }
}

pub struct GatherIndices<'a> {
    vals: &'a [i64],
    dims: &'a [usize],
}

impl<'a> GatherIndices<'a> {
    pub const fn new(vals: &'a [i64], dims: &'a [usize]) -> Self {
        Self { vals, dims }
    }

    pub const fn vals(&self) -> &'a [i64] {
        self.vals
    }

    pub const fn dims(&self) -> &'a [usize] {
        self.dims
    }
}

pub fn slice_bytes(src: ByteTensorView<'_>, slices: &[Slice]) -> (Shape, Vec<u8>) {
    let (bytes, dims, elem_size) = (src.bytes(), src.dims(), src.elem_size());
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

pub fn slice_assign_bytes(dst: ByteTensorView<'_>, slices: &[Slice], val_bytes: &[u8]) -> Vec<u8> {
    let (bytes, dims, elem_size) = (dst.bytes(), dst.dims(), dst.elem_size());
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

pub fn select_bytes(src: ByteTensorView<'_>, dim: usize, indices: &[i64]) -> (Shape, Vec<u8>) {
    let (bytes, dims, elem_size) = (src.bytes(), src.dims(), src.elem_size());
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
    src: ByteTensorView<'_>,
    indices: GatherIndices<'_>,
) -> (Shape, Vec<u8>) {
    let (bytes, dims, elem_size) = (src.bytes(), src.dims(), src.elem_size());
    let (idx_vals, idx_dims) = (indices.vals(), indices.dims());
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

pub fn mask_where_bytes(target: MaskedBytes<'_>, val_bytes: &[u8]) -> Vec<u8> {
    let (bytes, mask_bytes) = (target.bytes(), target.mask());
    let total = mask_bytes.len();
    let elem_size = bytes.len().checked_div(total).unwrap_or(0);
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

pub fn mask_fill_bytes(target: MaskedBytes<'_>, fill_elem: &[u8]) -> Vec<u8> {
    let (bytes, mask_bytes) = (target.bytes(), target.mask());
    let total = mask_bytes.len();
    let elem_size = fill_elem.len();
    let mut out = bytes.to_vec();
    for i in 0..total {
        if mask_bytes[i] != 0 {
            out[i * elem_size..(i + 1) * elem_size].copy_from_slice(fill_elem);
        }
    }
    out
}

pub fn unfold_bytes(
    src: ByteTensorView<'_>,
    params: crate::backend::UnfoldParams,
) -> (Shape, Vec<u8>) {
    let (bytes, dims, elem_size) = (src.bytes(), src.dims(), src.elem_size());
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

fn apply_update_f32(dst: &mut f32, val: f32, op: burn_backend::tensor::IndexingUpdateOp) {
    use burn_backend::tensor::IndexingUpdateOp;
    match op {
        IndexingUpdateOp::Assign => *dst = val,
        IndexingUpdateOp::Add => *dst += val,
        IndexingUpdateOp::Mul => *dst *= val,
        IndexingUpdateOp::Min => *dst = dst.min(val),
        IndexingUpdateOp::Max => *dst = dst.max(val),
    }
}

fn apply_update_i64(dst: &mut i64, val: i64, op: burn_backend::tensor::IndexingUpdateOp) {
    use burn_backend::tensor::IndexingUpdateOp;
    match op {
        IndexingUpdateOp::Assign => *dst = val,
        IndexingUpdateOp::Add => *dst = dst.wrapping_add(val),
        IndexingUpdateOp::Mul => *dst = dst.wrapping_mul(val),
        IndexingUpdateOp::Min => *dst = (*dst).min(val),
        IndexingUpdateOp::Max => *dst = (*dst).max(val),
    }
}

pub fn scatter_op_f32(
    dim: usize,
    tensor: GnaFloatTensorPrimitive,
    indices: GnaIntTensorPrimitive,
    value: GnaFloatTensorPrimitive,
    update: burn_backend::tensor::IndexingUpdateOp,
) -> GnaFloatTensorPrimitive {
    let shape = tensor.shape();
    let dtype = tensor.dtype();
    let mut out = decode_floats(tensor.buffer().as_slice(), dtype);
    let idx_vals = decode_ints(indices.buffer().as_slice(), indices.dtype());
    let val_vals = decode_floats(value.buffer().as_slice(), value.dtype());
    let out_strides = compute_strides(&shape);
    let idx_shape = indices.shape();
    for i in 0..idx_vals.len() {
        let mut coords = linear_to_coords(i, &idx_shape);
        coords[dim] = idx_vals[i] as usize;
        let dst_idx = coords_to_linear(&coords, &out_strides);
        apply_update_f32(&mut out[dst_idx], val_vals[i], update);
    }
    GnaFloatTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub fn scatter_op_i64(
    dim: usize,
    tensor: GnaIntTensorPrimitive,
    indices: GnaIntTensorPrimitive,
    value: GnaIntTensorPrimitive,
    update: burn_backend::tensor::IndexingUpdateOp,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dtype = tensor.dtype();
    let mut out = decode_ints(tensor.buffer().as_slice(), dtype);
    let idx_vals = decode_ints(indices.buffer().as_slice(), indices.dtype());
    let val_vals = decode_ints(value.buffer().as_slice(), value.dtype());
    let out_strides = compute_strides(&shape);
    let idx_shape = indices.shape();
    for i in 0..idx_vals.len() {
        let mut coords = linear_to_coords(i, &idx_shape);
        coords[dim] = idx_vals[i] as usize;
        let dst_idx = coords_to_linear(&coords, &out_strides);
        apply_update_i64(&mut out[dst_idx], val_vals[i], update);
    }
    GnaIntTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_ints(&out, dtype)),
    )
}

pub fn scatter_or_bool(
    dim: usize,
    tensor: GnaBoolTensorPrimitive,
    indices: GnaIntTensorPrimitive,
    value: GnaBoolTensorPrimitive,
) -> GnaBoolTensorPrimitive {
    let shape = tensor.shape();
    let mut out = tensor.buffer().to_vec();
    let idx_vals = decode_ints(indices.buffer().as_slice(), indices.dtype());
    let val_bytes = value.buffer().as_slice();
    let out_strides = compute_strides(&shape);
    let idx_shape = indices.shape();
    for i in 0..idx_vals.len() {
        let mut coords = linear_to_coords(i, &idx_shape);
        coords[dim] = idx_vals[i] as usize;
        let dst_idx = coords_to_linear(&coords, &out_strides);
        out[dst_idx] = u8::from(out[dst_idx] != 0 || val_bytes[i] != 0);
    }
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(out))
}

pub fn select_assign_op_f32(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
    indices: GnaIntTensorPrimitive,
    value: GnaFloatTensorPrimitive,
    update: burn_backend::tensor::IndexingUpdateOp,
) -> GnaFloatTensorPrimitive {
    let shape = tensor.shape();
    let dtype = tensor.dtype();
    let mut out = decode_floats(tensor.buffer().as_slice(), dtype);
    let idx_vals = decode_ints(indices.buffer().as_slice(), indices.dtype());
    let val_vals = decode_floats(value.buffer().as_slice(), value.dtype());
    let out_strides = compute_strides(&shape);
    let val_shape = value.shape();
    for (i, &val) in val_vals.iter().enumerate() {
        let mut coords = linear_to_coords(i, &val_shape);
        coords[dim] = idx_vals[coords[dim]] as usize;
        let dst_idx = coords_to_linear(&coords, &out_strides);
        apply_update_f32(&mut out[dst_idx], val, update);
    }
    GnaFloatTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub fn select_assign_op_i64(
    tensor: GnaIntTensorPrimitive,
    dim: usize,
    indices: GnaIntTensorPrimitive,
    value: GnaIntTensorPrimitive,
    update: burn_backend::tensor::IndexingUpdateOp,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dtype = tensor.dtype();
    let mut out = decode_ints(tensor.buffer().as_slice(), dtype);
    let idx_vals = decode_ints(indices.buffer().as_slice(), indices.dtype());
    let val_vals = decode_ints(value.buffer().as_slice(), value.dtype());
    let out_strides = compute_strides(&shape);
    let val_shape = value.shape();
    for (i, &val) in val_vals.iter().enumerate() {
        let mut coords = linear_to_coords(i, &val_shape);
        coords[dim] = idx_vals[coords[dim]] as usize;
        let dst_idx = coords_to_linear(&coords, &out_strides);
        apply_update_i64(&mut out[dst_idx], val, update);
    }
    GnaIntTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_ints(&out, dtype)),
    )
}

pub fn select_or_bool(
    tensor: GnaBoolTensorPrimitive,
    dim: usize,
    indices: GnaIntTensorPrimitive,
    value: GnaBoolTensorPrimitive,
) -> GnaBoolTensorPrimitive {
    let shape = tensor.shape();
    let mut out = tensor.buffer().to_vec();
    let idx_vals = decode_ints(indices.buffer().as_slice(), indices.dtype());
    let val_bytes = value.buffer().as_slice();
    let out_strides = compute_strides(&shape);
    let val_shape = value.shape();
    for (i, &vb) in val_bytes.iter().enumerate() {
        let mut coords = linear_to_coords(i, &val_shape);
        coords[dim] = idx_vals[coords[dim]] as usize;
        let dst_idx = coords_to_linear(&coords, &out_strides);
        out[dst_idx] = u8::from(out[dst_idx] != 0 || vb != 0);
    }
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(out))
}
