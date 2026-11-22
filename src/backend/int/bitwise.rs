use crate::tensor::GnaIntTensorPrimitive;
use burn_backend::Scalar;

pub(super) fn bitwise_and(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    lhs
}

pub(super) fn bitwise_and_scalar(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
) -> GnaIntTensorPrimitive {
    lhs
}

pub(super) fn bitwise_or(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    lhs
}

pub(super) fn bitwise_or_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
    lhs
}

pub(super) fn bitwise_xor(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    lhs
}

pub(super) fn bitwise_xor_scalar(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
) -> GnaIntTensorPrimitive {
    lhs
}

pub(super) fn bitwise_not(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn bitwise_left_shift(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    lhs
}

pub(super) fn bitwise_left_shift_scalar(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
) -> GnaIntTensorPrimitive {
    lhs
}

pub(super) fn bitwise_right_shift(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    lhs
}

pub(super) fn bitwise_right_shift_scalar(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
) -> GnaIntTensorPrimitive {
    lhs
}
