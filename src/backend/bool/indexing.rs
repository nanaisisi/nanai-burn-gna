use crate::backend::UnfoldParams;
use crate::backend::gna_exec::{decode_ints, layout};
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::{Scalar, Slice, TensorMetadata};

pub(super) fn bool_slice(
    tensor: GnaBoolTensorPrimitive,
    slices: &[Slice],
) -> GnaBoolTensorPrimitive {
    let (shape, bytes) =
        layout::slice_bytes(tensor.buffer().as_slice(), &tensor.shape(), slices, 1);
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
}

pub(super) fn bool_slice_assign(
    tensor: GnaBoolTensorPrimitive,
    slices: &[Slice],
    value: GnaBoolTensorPrimitive,
) -> GnaBoolTensorPrimitive {
    let bytes = layout::slice_assign_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        slices,
        value.buffer().as_slice(),
        1,
    );
    GnaBoolTensorPrimitive::new(tensor.shape(), GnaBuffer::from_bytes(bytes))
}

pub(super) fn bool_mask_where(
    tensor: GnaBoolTensorPrimitive,
    mask: GnaBoolTensorPrimitive,
    value: GnaBoolTensorPrimitive,
) -> GnaBoolTensorPrimitive {
    let bytes = layout::mask_where_bytes(
        tensor.buffer().as_slice(),
        mask.buffer().as_slice(),
        value.buffer().as_slice(),
        1,
    );
    GnaBoolTensorPrimitive::new(tensor.shape(), GnaBuffer::from_bytes(bytes))
}

pub(super) fn bool_mask_fill(
    tensor: GnaBoolTensorPrimitive,
    mask: GnaBoolTensorPrimitive,
    value: Scalar,
) -> GnaBoolTensorPrimitive {
    let fill = [u8::from(value.elem::<bool>())];
    let bytes = layout::mask_fill_bytes(
        tensor.buffer().as_slice(),
        mask.buffer().as_slice(),
        &fill,
        1,
    );
    GnaBoolTensorPrimitive::new(tensor.shape(), GnaBuffer::from_bytes(bytes))
}

pub(super) fn bool_gather(
    dim: usize,
    tensor: GnaBoolTensorPrimitive,
    indices: GnaIntTensorPrimitive,
) -> GnaBoolTensorPrimitive {
    let idx_vals = decode_ints(indices.buffer().as_slice(), indices.dtype());
    let (shape, bytes) = layout::gather_bytes(
        dim,
        tensor.buffer().as_slice(),
        &tensor.shape(),
        &idx_vals,
        &indices.shape(),
        1,
    );
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
}

pub(super) fn bool_scatter_or(
    dim: usize,
    tensor: GnaBoolTensorPrimitive,
    indices: GnaIntTensorPrimitive,
    value: GnaBoolTensorPrimitive,
) -> GnaBoolTensorPrimitive {
    layout::scatter_or_bool(dim, tensor, indices, value)
}

pub(super) fn bool_select(
    tensor: GnaBoolTensorPrimitive,
    dim: usize,
    indices: GnaIntTensorPrimitive,
) -> GnaBoolTensorPrimitive {
    let idx_vals = decode_ints(indices.buffer().as_slice(), indices.dtype());
    let (shape, bytes) = layout::select_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        dim,
        &idx_vals,
        1,
    );
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
}

pub(super) fn bool_select_or(
    tensor: GnaBoolTensorPrimitive,
    dim: usize,
    indices: GnaIntTensorPrimitive,
    value: GnaBoolTensorPrimitive,
) -> GnaBoolTensorPrimitive {
    layout::select_or_bool(tensor, dim, indices, value)
}

pub(super) fn bool_swap_dims(
    tensor: GnaBoolTensorPrimitive,
    dim1: usize,
    dim2: usize,
) -> GnaBoolTensorPrimitive {
    let (shape, bytes) =
        layout::swap_dims_bytes(tensor.buffer().as_slice(), &tensor.shape(), dim1, dim2, 1);
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
}

pub(super) fn bool_permute(
    tensor: GnaBoolTensorPrimitive,
    axes: &[usize],
) -> GnaBoolTensorPrimitive {
    let (shape, bytes) =
        layout::permute_bytes(tensor.buffer().as_slice(), &tensor.shape(), axes, 1);
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
}

pub(super) fn bool_flip(tensor: GnaBoolTensorPrimitive, axes: &[usize]) -> GnaBoolTensorPrimitive {
    let bytes = layout::flip_bytes(tensor.buffer().as_slice(), &tensor.shape(), axes, 1);
    GnaBoolTensorPrimitive::new(tensor.shape(), GnaBuffer::from_bytes(bytes))
}

pub(super) fn bool_unfold(
    tensor: GnaBoolTensorPrimitive,
    dim: usize,
    size: usize,
    step: usize,
) -> GnaBoolTensorPrimitive {
    let (shape, bytes) = layout::unfold_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        UnfoldParams::new(dim, size, step),
        1,
    );
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
}
