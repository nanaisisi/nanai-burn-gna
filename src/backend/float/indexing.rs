use crate::backend::UnfoldParams;
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::tensor::IndexingUpdateOp;
use burn_backend::{Scalar, Slice};

pub(super) fn float_swap_dims(
    tensor: GnaFloatTensorPrimitive,
    _dim1: usize,
    _dim2: usize,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_permute(
    tensor: GnaFloatTensorPrimitive,
    _axes: &[usize],
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_flip(
    tensor: GnaFloatTensorPrimitive,
    _axes: &[usize],
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_gather(
    _dim: usize,
    tensor: GnaFloatTensorPrimitive,
    _indices: GnaIntTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_scatter(
    _dim: usize,
    tensor: GnaFloatTensorPrimitive,
    _indices: GnaIntTensorPrimitive,
    _value: GnaFloatTensorPrimitive,
    _update: IndexingUpdateOp,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_select(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
    _indices: GnaIntTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_select_assign(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
    _indices: GnaIntTensorPrimitive,
    _value: GnaFloatTensorPrimitive,
    _update: IndexingUpdateOp,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_slice(
    tensor: GnaFloatTensorPrimitive,
    _slices: &[Slice],
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_slice_assign(
    tensor: GnaFloatTensorPrimitive,
    _slices: &[Slice],
    _value: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_mask_where(
    tensor: GnaFloatTensorPrimitive,
    _mask: GnaBoolTensorPrimitive,
    _value: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_mask_fill(
    tensor: GnaFloatTensorPrimitive,
    _mask: GnaBoolTensorPrimitive,
    _value: Scalar,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_unfold(
    tensor: GnaFloatTensorPrimitive,
    _params: UnfoldParams,
) -> GnaFloatTensorPrimitive {
    tensor
}
