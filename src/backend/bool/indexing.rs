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
    let (shape, bytes) = layout::slice_bytes(
        layout::shape::ByteTensorView::new(tensor.buffer().as_slice(), &tensor.shape(), 1),
        slices,
    );
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
}

pub(super) fn bool_slice_assign(
    tensor: GnaBoolTensorPrimitive,
    slices: &[Slice],
    value: GnaBoolTensorPrimitive,
) -> GnaBoolTensorPrimitive {
    let bytes = layout::slice_assign_bytes(
        layout::shape::ByteTensorView::new(tensor.buffer().as_slice(), &tensor.shape(), 1),
        slices,
        value.buffer().as_slice(),
    );
    GnaBoolTensorPrimitive::new(tensor.shape(), GnaBuffer::from_bytes(bytes))
}

pub(super) fn bool_mask_where(
    tensor: GnaBoolTensorPrimitive,
    mask: GnaBoolTensorPrimitive,
    value: GnaBoolTensorPrimitive,
) -> GnaBoolTensorPrimitive {
    let bytes = layout::mask_where_bytes(
        layout::slice::MaskedBytes {
            bytes: tensor.buffer().as_slice(),
            mask: mask.buffer().as_slice(),
        },
        value.buffer().as_slice(),
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
        layout::slice::MaskedBytes {
            bytes: tensor.buffer().as_slice(),
            mask: mask.buffer().as_slice(),
        },
        &fill,
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
        layout::shape::ByteTensorView::new(tensor.buffer().as_slice(), &tensor.shape(), 1),
        layout::slice::GatherIndices {
            vals: &idx_vals,
            dims: &indices.shape(),
        },
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
        layout::shape::ByteTensorView::new(tensor.buffer().as_slice(), &tensor.shape(), 1),
        dim,
        &idx_vals,
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
    let (shape, bytes) = layout::swap_dims_bytes(
        layout::shape::ByteTensorView::new(tensor.buffer().as_slice(), &tensor.shape(), 1),
        dim1,
        dim2,
    );
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
}

pub(super) fn bool_permute(
    tensor: GnaBoolTensorPrimitive,
    axes: &[usize],
) -> GnaBoolTensorPrimitive {
    let (shape, bytes) = layout::permute_bytes(
        layout::shape::ByteTensorView::new(tensor.buffer().as_slice(), &tensor.shape(), 1),
        axes,
    );
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
}

pub(super) fn bool_flip(tensor: GnaBoolTensorPrimitive, axes: &[usize]) -> GnaBoolTensorPrimitive {
    let bytes = layout::flip_bytes(
        layout::shape::ByteTensorView::new(tensor.buffer().as_slice(), &tensor.shape(), 1),
        axes,
    );
    GnaBoolTensorPrimitive::new(tensor.shape(), GnaBuffer::from_bytes(bytes))
}

pub(super) fn bool_unfold(
    tensor: GnaBoolTensorPrimitive,
    params: UnfoldParams,
) -> GnaBoolTensorPrimitive {
    let (shape, bytes) = layout::unfold_bytes(
        layout::shape::ByteTensorView::new(tensor.buffer().as_slice(), &tensor.shape(), 1),
        params,
    );
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
}
