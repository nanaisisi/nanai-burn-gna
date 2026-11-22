use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::IntDType;

pub(super) fn float_sum(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::ops::execute_float_sum(tensor)
}

pub(super) fn float_sum_dim(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::ops::execute_float_sum_dim(tensor, dim, false)
}

pub(super) fn float_mean_dim(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::ops::execute_float_sum_dim(tensor, dim, true)
}

pub(super) fn float_cumsum(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_cumprod(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_cummin(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_cummax(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_argmax(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
    out_dtype: IntDType,
) -> GnaIntTensorPrimitive {
    let (shape, _dtype, buffer) = tensor.into_parts();
    GnaIntTensorPrimitive::new(shape, out_dtype, buffer)
}

pub(super) fn float_argtopk(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
    _k: usize,
    out_dtype: IntDType,
) -> GnaIntTensorPrimitive {
    let (shape, _dtype, buffer) = tensor.into_parts();
    GnaIntTensorPrimitive::new(shape, out_dtype, buffer)
}

pub(super) fn float_argmin(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
    out_dtype: IntDType,
) -> GnaIntTensorPrimitive {
    let (shape, _dtype, buffer) = tensor.into_parts();
    GnaIntTensorPrimitive::new(shape, out_dtype, buffer)
}
