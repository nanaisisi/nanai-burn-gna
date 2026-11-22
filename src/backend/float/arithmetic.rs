use crate::tensor::float::GnaFloatTensorPrimitive;
use burn_backend::Scalar;

pub(super) fn float_add(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::ops::execute_float_add_sub(lhs, rhs, 1)
}

pub(super) fn float_add_scalar(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::ops::execute_float_add_scalar(lhs, rhs, 1.0)
}

pub(super) fn float_sub(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::ops::execute_float_add_sub(lhs, rhs, -1)
}

pub(super) fn float_sub_scalar(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::ops::execute_float_add_scalar(lhs, rhs, -1.0)
}

pub(super) fn float_mul(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::ops::execute_float_mul(lhs, rhs)
}

pub(super) fn float_mul_scalar(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::ops::execute_float_mul_scalar(lhs, rhs)
}

pub(super) fn float_div(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::extra::map_float_binary(lhs, rhs, |a, b| a / b)
}

pub(super) fn float_div_scalar(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::extra::map_float_scalar(lhs, rhs, |a, b| a / b)
}

pub(super) fn float_remainder(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::extra::map_float_binary(lhs, rhs, |a, b| a % b)
}

pub(super) fn float_remainder_scalar(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::extra::map_float_scalar(lhs, rhs, |a, b| a % b)
}

pub(super) fn float_matmul(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::execute_float_matmul(lhs, rhs)
}

pub(super) fn float_cross(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
    dim: usize,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::extra::execute_float_cross(lhs, rhs, dim)
}

pub(super) fn float_recip(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::extra::map_float_unary(tensor, |a| 1.0 / a)
}

pub(super) fn float_powf(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::extra::map_float_binary(lhs, rhs, f32::powf)
}

pub(super) fn float_powf_scalar_impl(
    tensor: GnaFloatTensorPrimitive,
    value: Scalar,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::extra::map_float_scalar(tensor, value, f32::powf)
}

pub(super) fn float_atan2(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::extra::map_float_binary(lhs, rhs, f32::atan2)
}
