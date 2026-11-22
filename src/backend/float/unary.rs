use crate::tensor::{GnaFloatTensorPrimitive, GnaIntTensorPrimitive};
use burn_backend::IntDType;

pub(super) fn float_sum(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_sum_dim(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_mean_dim(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
) -> GnaFloatTensorPrimitive {
    tensor
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

pub(super) fn float_exp(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_log(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_log1p(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_sqrt(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_abs(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_cos(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_sin(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_tan(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_cosh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_sinh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_tanh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_acos(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_acosh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_asin(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_asinh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_atan(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_atanh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_round(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_floor(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_ceil(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_trunc(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_erf(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_argmax(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
    out_dtype: IntDType,
) -> GnaIntTensorPrimitive {
    GnaIntTensorPrimitive {
        shape: tensor.shape,
        dtype: out_dtype,
        buffer: tensor.buffer,
    }
}

pub(super) fn float_argtopk(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
    _k: usize,
    out_dtype: IntDType,
) -> GnaIntTensorPrimitive {
    GnaIntTensorPrimitive {
        shape: tensor.shape,
        dtype: out_dtype,
        buffer: tensor.buffer,
    }
}

pub(super) fn float_argmin(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
    out_dtype: IntDType,
) -> GnaIntTensorPrimitive {
    GnaIntTensorPrimitive {
        shape: tensor.shape,
        dtype: out_dtype,
        buffer: tensor.buffer,
    }
}

pub(super) fn float_unfold(
    tensor: GnaFloatTensorPrimitive,
    _dim: usize,
    _size: usize,
    _step: usize,
) -> GnaFloatTensorPrimitive {
    tensor
}
