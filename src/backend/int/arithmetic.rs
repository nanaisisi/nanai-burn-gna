use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::Scalar;

pub(super) fn int_add(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    crate::backend::gna_exec::ops::execute_int_add_sub(lhs, rhs, 1)
}

pub(super) fn int_add_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
    crate::backend::gna_exec::ops::execute_int_add_scalar(lhs, rhs, 1)
}

pub(super) fn int_sub(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    crate::backend::gna_exec::ops::execute_int_add_sub(lhs, rhs, -1)
}

pub(super) fn int_sub_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
    crate::backend::gna_exec::ops::execute_int_add_scalar(lhs, rhs, -1)
}

pub(super) fn int_mul(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    crate::backend::gna_exec::ops::execute_int_mul(lhs, rhs)
}

pub(super) fn int_mul_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
    crate::backend::gna_exec::ops::execute_int_mul_scalar(lhs, rhs)
}

pub(super) fn int_div(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    lhs
}

pub(super) fn int_div_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
    lhs
}

pub(super) fn int_remainder(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    lhs
}

pub(super) fn int_remainder_scalar(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
) -> GnaIntTensorPrimitive {
    lhs
}

pub(super) fn int_matmul(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    crate::backend::gna_exec::execute_int_matmul(lhs, rhs)
}
