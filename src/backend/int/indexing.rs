use crate::backend::UnfoldParams;
use crate::backend::gna_exec::decode_ints;
use crate::backend::gna_exec::layout;
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::tensor::IndexingUpdateOp;
use burn_backend::{DType, Scalar, Slice, TensorMetadata};

pub(super) fn int_slice(tensor: GnaIntTensorPrimitive, slices: &[Slice]) -> GnaIntTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let (out_shape, out_bytes) = layout::slice_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        slices,
        elem_size,
    );
    GnaIntTensorPrimitive::new(out_shape, tensor.dtype(), GnaBuffer::from_bytes(out_bytes))
}

pub(super) fn int_slice_assign(
    tensor: GnaIntTensorPrimitive,
    slices: &[Slice],
    value: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let out_bytes = layout::slice_assign_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        slices,
        value.buffer().as_slice(),
        elem_size,
    );
    GnaIntTensorPrimitive::new(
        tensor.shape(),
        tensor.dtype(),
        GnaBuffer::from_bytes(out_bytes),
    )
}

pub(super) fn int_mask_where(
    tensor: GnaIntTensorPrimitive,
    mask: GnaBoolTensorPrimitive,
    value: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let out_bytes = layout::mask_where_bytes(
        tensor.buffer().as_slice(),
        mask.buffer().as_slice(),
        value.buffer().as_slice(),
        elem_size,
    );
    GnaIntTensorPrimitive::new(
        tensor.shape(),
        tensor.dtype(),
        GnaBuffer::from_bytes(out_bytes),
    )
}

pub(super) fn int_mask_fill(
    tensor: GnaIntTensorPrimitive,
    mask: GnaBoolTensorPrimitive,
    value: Scalar,
) -> GnaIntTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let fill = layout::int_scalar_to_le_bytes(value, elem_size);
    let out_bytes = layout::mask_fill_bytes(
        tensor.buffer().as_slice(),
        mask.buffer().as_slice(),
        &fill,
        elem_size,
    );
    GnaIntTensorPrimitive::new(
        tensor.shape(),
        tensor.dtype(),
        GnaBuffer::from_bytes(out_bytes),
    )
}

pub(super) fn int_gather(
    dim: usize,
    tensor: GnaIntTensorPrimitive,
    indices: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let idx_vals = decode_ints(indices.buffer().as_slice(), indices.dtype());
    let (out_shape, out_bytes) = layout::gather_bytes(
        dim,
        tensor.buffer().as_slice(),
        &tensor.shape(),
        &idx_vals,
        &indices.shape(),
        elem_size,
    );
    GnaIntTensorPrimitive::new(out_shape, tensor.dtype(), GnaBuffer::from_bytes(out_bytes))
}

pub(super) fn int_scatter(
    _dim: usize,
    tensor: GnaIntTensorPrimitive,
    _indices: GnaIntTensorPrimitive,
    _value: GnaIntTensorPrimitive,
    _update: IndexingUpdateOp,
) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_select(
    tensor: GnaIntTensorPrimitive,
    dim: usize,
    indices: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let idx_vals = decode_ints(indices.buffer().as_slice(), indices.dtype());
    let (out_shape, out_bytes) = layout::select_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        dim,
        &idx_vals,
        elem_size,
    );
    GnaIntTensorPrimitive::new(out_shape, tensor.dtype(), GnaBuffer::from_bytes(out_bytes))
}

pub(super) fn int_select_assign(
    tensor: GnaIntTensorPrimitive,
    _dim: usize,
    _indices: GnaIntTensorPrimitive,
    _value: GnaIntTensorPrimitive,
    _update: IndexingUpdateOp,
) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_swap_dims(
    tensor: GnaIntTensorPrimitive,
    dim1: usize,
    dim2: usize,
) -> GnaIntTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let (out_shape, out_bytes) = layout::swap_dims_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        dim1,
        dim2,
        elem_size,
    );
    GnaIntTensorPrimitive::new(out_shape, tensor.dtype(), GnaBuffer::from_bytes(out_bytes))
}

pub(super) fn int_permute(tensor: GnaIntTensorPrimitive, axes: &[usize]) -> GnaIntTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let (out_shape, out_bytes) =
        layout::permute_bytes(tensor.buffer().as_slice(), &tensor.shape(), axes, elem_size);
    GnaIntTensorPrimitive::new(out_shape, tensor.dtype(), GnaBuffer::from_bytes(out_bytes))
}

pub(super) fn int_flip(tensor: GnaIntTensorPrimitive, axes: &[usize]) -> GnaIntTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let out_bytes =
        layout::flip_bytes(tensor.buffer().as_slice(), &tensor.shape(), axes, elem_size);
    GnaIntTensorPrimitive::new(
        tensor.shape(),
        tensor.dtype(),
        GnaBuffer::from_bytes(out_bytes),
    )
}

pub(super) fn int_unfold(
    tensor: GnaIntTensorPrimitive,
    params: UnfoldParams,
) -> GnaIntTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let (out_shape, out_bytes) = layout::unfold_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        params,
        elem_size,
    );
    GnaIntTensorPrimitive::new(out_shape, tensor.dtype(), GnaBuffer::from_bytes(out_bytes))
}
