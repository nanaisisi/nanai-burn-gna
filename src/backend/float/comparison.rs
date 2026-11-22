use crate::backend::gna_exec::layout::{CmpOp, compare_float, compare_float_elem};
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::float::GnaFloatTensorPrimitive;
use burn_backend::{BoolStore, Scalar};

pub(super) fn float_equal(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_float(lhs, rhs, CmpOp::Equal)
}

pub(super) fn float_equal_elem(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_float_elem(lhs, rhs, CmpOp::Equal)
}

pub(super) fn float_greater(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_float(lhs, rhs, CmpOp::Greater)
}

pub(super) fn float_greater_elem(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_float_elem(lhs, rhs, CmpOp::Greater)
}

pub(super) fn float_greater_equal(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_float(lhs, rhs, CmpOp::GreaterEqual)
}

pub(super) fn float_greater_equal_elem(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_float_elem(lhs, rhs, CmpOp::GreaterEqual)
}

pub(super) fn float_lower(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_float(lhs, rhs, CmpOp::Lower)
}

pub(super) fn float_lower_elem(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_float_elem(lhs, rhs, CmpOp::Lower)
}

pub(super) fn float_lower_equal(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_float(lhs, rhs, CmpOp::LowerEqual)
}

pub(super) fn float_lower_equal_elem(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    compare_float_elem(lhs, rhs, CmpOp::LowerEqual)
}
