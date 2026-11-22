use crate::backend::gna_exec::layout::{CmpOp, compare_int, compare_int_elem};
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::{BoolStore, Scalar};

pub(super) fn int_equal(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_int(lhs, rhs, CmpOp::Equal)
}

pub(super) fn int_equal_elem(
    lhs: GnaIntTensorPrimitive,
    rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_int_elem(lhs, rhs, CmpOp::Equal)
}

pub(super) fn int_greater(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_int(lhs, rhs, CmpOp::Greater)
}

pub(super) fn int_greater_elem(
    lhs: GnaIntTensorPrimitive,
    rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_int_elem(lhs, rhs, CmpOp::Greater)
}

pub(super) fn int_greater_equal(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_int(lhs, rhs, CmpOp::GreaterEqual)
}

pub(super) fn int_greater_equal_elem(
    lhs: GnaIntTensorPrimitive,
    rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_int_elem(lhs, rhs, CmpOp::GreaterEqual)
}

pub(super) fn int_lower(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_int(lhs, rhs, CmpOp::Lower)
}

pub(super) fn int_lower_elem(
    lhs: GnaIntTensorPrimitive,
    rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_int_elem(lhs, rhs, CmpOp::Lower)
}

pub(super) fn int_lower_equal(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_int(lhs, rhs, CmpOp::LowerEqual)
}

pub(super) fn int_lower_equal_elem(
    lhs: GnaIntTensorPrimitive,
    rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_int_elem(lhs, rhs, CmpOp::LowerEqual)
}
