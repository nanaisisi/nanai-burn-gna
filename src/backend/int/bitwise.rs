use crate::backend::gna_exec::extra::{map_int_binary, map_int_scalar, map_int_unary};
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::Scalar;

pub(super) fn bitwise_and(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    map_int_binary(lhs, rhs, |a, b| a & b)
}

pub(super) fn bitwise_and_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
    map_int_scalar(lhs, rhs, |a, b| a & b)
}

pub(super) fn bitwise_or(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    map_int_binary(lhs, rhs, |a, b| a | b)
}

pub(super) fn bitwise_or_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
    map_int_scalar(lhs, rhs, |a, b| a | b)
}

pub(super) fn bitwise_xor(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    map_int_binary(lhs, rhs, |a, b| a ^ b)
}

pub(super) fn bitwise_xor_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
    map_int_scalar(lhs, rhs, |a, b| a ^ b)
}

pub(super) fn bitwise_not(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
    map_int_unary(tensor, |a| !a)
}

pub(super) fn bitwise_left_shift(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    map_int_binary(lhs, rhs, |a, b| a.wrapping_shl(b as u32))
}

pub(super) fn bitwise_left_shift_scalar(
    lhs: GnaIntTensorPrimitive,
    rhs: Scalar,
) -> GnaIntTensorPrimitive {
    map_int_scalar(lhs, rhs, |a, b| a.wrapping_shl(b as u32))
}

pub(super) fn bitwise_right_shift(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    map_int_binary(lhs, rhs, |a, b| a.wrapping_shr(b as u32))
}

pub(super) fn bitwise_right_shift_scalar(
    lhs: GnaIntTensorPrimitive,
    rhs: Scalar,
) -> GnaIntTensorPrimitive {
    map_int_scalar(lhs, rhs, |a, b| a.wrapping_shr(b as u32))
}
