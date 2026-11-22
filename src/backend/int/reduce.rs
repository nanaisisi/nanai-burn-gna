use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::IntDType;

pub(super) fn int_sum(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
    crate::backend::gna_exec::ops::execute_int_sum(tensor)
}

pub(super) fn int_sum_dim(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
    crate::backend::gna_exec::ops::execute_int_sum_dim(tensor, dim, false)
}

pub(super) fn int_prod(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_prod_dim(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_mean_dim(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
    crate::backend::gna_exec::ops::execute_int_sum_dim(tensor, dim, true)
}

pub(super) fn int_cumsum(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_cumprod(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_cummin(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_cummax(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_argmax(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
    tensor.with_dtype(IntDType::I32)
}

pub(super) fn int_argtopk(
    tensor: GnaIntTensorPrimitive,
    _dim: usize,
    _k: usize,
) -> GnaIntTensorPrimitive {
    tensor.with_dtype(IntDType::I32)
}

pub(super) fn int_argmin(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
    tensor.with_dtype(IntDType::I32)
}
