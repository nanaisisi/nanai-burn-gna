use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::{BoolStore, Scalar};

fn into_bool(tensor: GnaIntTensorPrimitive) -> GnaBoolTensorPrimitive {
    let (shape, _dtype, buffer) = tensor.into_parts();
    GnaBoolTensorPrimitive::new(shape, buffer)
}

pub(super) fn int_equal(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn int_equal_elem(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn int_greater(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn int_greater_elem(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn int_greater_equal(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn int_greater_equal_elem(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn int_lower(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn int_lower_elem(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn int_lower_equal(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}

pub(super) fn int_lower_equal_elem(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    into_bool(lhs)
}
