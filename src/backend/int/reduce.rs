use crate::backend::gna_exec::extra::{
    CumMode, execute_int_argmax_argmin, execute_int_argtopk, execute_int_cum, execute_int_prod,
    execute_int_prod_dim,
};
use crate::tensor::int::GnaIntTensorPrimitive;

pub(super) fn int_sum(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
    crate::backend::gna_exec::ops::execute_int_sum(tensor)
}

pub(super) fn int_sum_dim(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
    crate::backend::gna_exec::ops::execute_int_sum_dim(tensor, dim, false)
}

pub(super) fn int_prod(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
    execute_int_prod(tensor)
}

pub(super) fn int_prod_dim(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
    execute_int_prod_dim(tensor, dim)
}

pub(super) fn int_mean_dim(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
    crate::backend::gna_exec::ops::execute_int_sum_dim(tensor, dim, true)
}

pub(super) fn int_cumsum(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
    execute_int_cum(tensor, dim, CumMode::Sum)
}

pub(super) fn int_cumprod(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
    execute_int_cum(tensor, dim, CumMode::Prod)
}

pub(super) fn int_cummin(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
    execute_int_cum(tensor, dim, CumMode::Min)
}

pub(super) fn int_cummax(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
    execute_int_cum(tensor, dim, CumMode::Max)
}

pub(super) fn int_argmax(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
    execute_int_argmax_argmin(tensor, dim, true)
}

pub(super) fn int_argtopk(
    tensor: GnaIntTensorPrimitive,
    dim: usize,
    k: usize,
) -> GnaIntTensorPrimitive {
    execute_int_argtopk(tensor, dim, k)
}

pub(super) fn int_argmin(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
    execute_int_argmax_argmin(tensor, dim, false)
}
