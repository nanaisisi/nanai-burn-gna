use crate::backend::UnfoldParams;
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::tensor::IndexingUpdateOp;
use burn_backend::{Scalar, Slice};

pub(super) fn int_slice(tensor: GnaIntTensorPrimitive, _slices: &[Slice]) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_slice_assign(
    tensor: GnaIntTensorPrimitive,
    _slices: &[Slice],
    _value: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_mask_where(
    tensor: GnaIntTensorPrimitive,
    _mask: GnaBoolTensorPrimitive,
    _value: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_mask_fill(
    tensor: GnaIntTensorPrimitive,
    _mask: GnaBoolTensorPrimitive,
    _value: Scalar,
) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_gather(
    _dim: usize,
    tensor: GnaIntTensorPrimitive,
    _indices: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    tensor
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
    _dim: usize,
    _indices: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    tensor
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
    _dim1: usize,
    _dim2: usize,
) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_permute(tensor: GnaIntTensorPrimitive, _axes: &[usize]) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_flip(tensor: GnaIntTensorPrimitive, _axes: &[usize]) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_unfold(
    tensor: GnaIntTensorPrimitive,
    _params: UnfoldParams,
) -> GnaIntTensorPrimitive {
    tensor
}
