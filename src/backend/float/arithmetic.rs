use crate::tensor::float::GnaFloatTensorPrimitive;
use burn_backend::Scalar;

pub(super) fn float_add(
    lhs: GnaFloatTensorPrimitive,
    _rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    lhs
}

pub(super) fn float_add_scalar(
    lhs: GnaFloatTensorPrimitive,
    _rhs: Scalar,
) -> GnaFloatTensorPrimitive {
    lhs
}

pub(super) fn float_sub(
    lhs: GnaFloatTensorPrimitive,
    _rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    lhs
}

pub(super) fn float_sub_scalar(
    lhs: GnaFloatTensorPrimitive,
    _rhs: Scalar,
) -> GnaFloatTensorPrimitive {
    lhs
}

pub(super) fn float_mul(
    lhs: GnaFloatTensorPrimitive,
    _rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    lhs
}

pub(super) fn float_mul_scalar(
    lhs: GnaFloatTensorPrimitive,
    _rhs: Scalar,
) -> GnaFloatTensorPrimitive {
    lhs
}

pub(super) fn float_div(
    lhs: GnaFloatTensorPrimitive,
    _rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    lhs
}

pub(super) fn float_div_scalar(
    lhs: GnaFloatTensorPrimitive,
    _rhs: Scalar,
) -> GnaFloatTensorPrimitive {
    lhs
}

pub(super) fn float_remainder(
    lhs: GnaFloatTensorPrimitive,
    _rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    lhs
}

pub(super) fn float_remainder_scalar(
    lhs: GnaFloatTensorPrimitive,
    _rhs: Scalar,
) -> GnaFloatTensorPrimitive {
    lhs
}

pub(super) fn float_matmul(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::execute_float_matmul(lhs, rhs)
}

pub(super) fn float_cross(
    lhs: GnaFloatTensorPrimitive,
    _rhs: GnaFloatTensorPrimitive,
    _dim: usize,
) -> GnaFloatTensorPrimitive {
    lhs
}

pub(super) fn float_recip(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_powf(
    lhs: GnaFloatTensorPrimitive,
    _rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    lhs
}

pub(super) fn float_powf_scalar_impl(
    tensor: GnaFloatTensorPrimitive,
    _value: Scalar,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_atan2(
    lhs: GnaFloatTensorPrimitive,
    _rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    lhs
}
