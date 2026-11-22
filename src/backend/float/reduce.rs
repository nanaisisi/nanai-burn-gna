use crate::backend::gna_exec::extra::{
    CumMode, execute_float_argmax_argmin, execute_float_argtopk, execute_float_cum,
};
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

pub(super) fn float_cumsum(tensor: GnaFloatTensorPrimitive, dim: usize) -> GnaFloatTensorPrimitive {
    execute_float_cum(tensor, dim, CumMode::Sum)
}

pub(super) fn float_cumprod(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
) -> GnaFloatTensorPrimitive {
    execute_float_cum(tensor, dim, CumMode::Prod)
}

pub(super) fn float_cummin(tensor: GnaFloatTensorPrimitive, dim: usize) -> GnaFloatTensorPrimitive {
    execute_float_cum(tensor, dim, CumMode::Min)
}

pub(super) fn float_cummax(tensor: GnaFloatTensorPrimitive, dim: usize) -> GnaFloatTensorPrimitive {
    execute_float_cum(tensor, dim, CumMode::Max)
}

pub(super) fn float_argmax(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
    out_dtype: IntDType,
) -> GnaIntTensorPrimitive {
    execute_float_argmax_argmin(tensor, dim, out_dtype, true)
}

pub(super) fn float_argtopk(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
    k: usize,
    out_dtype: IntDType,
) -> GnaIntTensorPrimitive {
    execute_float_argtopk(tensor, dim, k, out_dtype)
}

pub(super) fn float_argmin(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
    out_dtype: IntDType,
) -> GnaIntTensorPrimitive {
    execute_float_argmax_argmin(tensor, dim, out_dtype, false)
}
