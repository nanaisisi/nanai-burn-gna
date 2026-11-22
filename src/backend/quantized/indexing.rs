use crate::backend::gna_exec::decode_ints;
use crate::backend::gna_exec::module_ops::{QLayoutOp, transform_q_bytes};
use crate::tensor::int::GnaIntTensorPrimitive;
use crate::tensor::quantized::GnaQTensorPrimitive;
use burn_backend::Slice;

pub(super) fn q_swap_dims(
    tensor: GnaQTensorPrimitive,
    dim1: usize,
    dim2: usize,
) -> GnaQTensorPrimitive {
    transform_q_bytes(tensor, QLayoutOp::SwapDims(dim1, dim2))
}

pub(super) fn q_permute(tensor: GnaQTensorPrimitive, axes: &[usize]) -> GnaQTensorPrimitive {
    transform_q_bytes(tensor, QLayoutOp::Permute(axes))
}

pub(super) fn q_flip(tensor: GnaQTensorPrimitive, axes: &[usize]) -> GnaQTensorPrimitive {
    transform_q_bytes(tensor, QLayoutOp::Flip(axes))
}

pub(super) fn q_select(
    tensor: GnaQTensorPrimitive,
    dim: usize,
    indices: GnaIntTensorPrimitive,
) -> GnaQTensorPrimitive {
    let idx = decode_ints(indices.buffer().as_slice(), indices.dtype());
    transform_q_bytes(tensor, QLayoutOp::Select(dim, &idx))
}

pub(super) fn q_slice(tensor: GnaQTensorPrimitive, slices: &[Slice]) -> GnaQTensorPrimitive {
    transform_q_bytes(tensor, QLayoutOp::Slice(slices))
}
