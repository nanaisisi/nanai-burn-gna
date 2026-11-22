use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::float::GnaFloatTensorPrimitive;
use burn_backend::{BoolStore, Scalar};

fn into_bool(tensor: GnaFloatTensorPrimitive) -> GnaBoolTensorPrimitive {
    let (shape, _dtype, buffer) = tensor.into_parts();
    GnaBoolTensorPrimitive::new(shape, buffer)
}

pub(super) fn float_equal(
    lhs: GnaFloatTensorPrimitive,
    _rhs: GnaFloatTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn float_equal_elem(
    lhs: GnaFloatTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn float_greater(
    lhs: GnaFloatTensorPrimitive,
    _rhs: GnaFloatTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn float_greater_elem(
    lhs: GnaFloatTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn float_greater_equal(
    lhs: GnaFloatTensorPrimitive,
    _rhs: GnaFloatTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn float_greater_equal_elem(
    lhs: GnaFloatTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn float_lower(
    lhs: GnaFloatTensorPrimitive,
    _rhs: GnaFloatTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn float_lower_elem(
    lhs: GnaFloatTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn float_lower_equal(
    lhs: GnaFloatTensorPrimitive,
    _rhs: GnaFloatTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn float_lower_equal_elem(
    lhs: GnaFloatTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}
