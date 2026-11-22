use crate::tensor::{GnaBoolTensorPrimitive, GnaIntTensorPrimitive};
use burn_backend::{BoolStore, Scalar};

pub(super) fn int_equal(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    GnaBoolTensorPrimitive {
        shape: lhs.shape,
        buffer: lhs.buffer,
    }
}

pub(super) fn int_equal_elem(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    GnaBoolTensorPrimitive {
        shape: lhs.shape,
        buffer: lhs.buffer,
    }
}

pub(super) fn int_greater(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    GnaBoolTensorPrimitive {
        shape: lhs.shape,
        buffer: lhs.buffer,
    }
}

pub(super) fn int_greater_elem(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    GnaBoolTensorPrimitive {
        shape: lhs.shape,
        buffer: lhs.buffer,
    }
}

pub(super) fn int_greater_equal(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    GnaBoolTensorPrimitive {
        shape: lhs.shape,
        buffer: lhs.buffer,
    }
}

pub(super) fn int_greater_equal_elem(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    GnaBoolTensorPrimitive {
        shape: lhs.shape,
        buffer: lhs.buffer,
    }
}

pub(super) fn int_lower(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    GnaBoolTensorPrimitive {
        shape: lhs.shape,
        buffer: lhs.buffer,
    }
}

pub(super) fn int_lower_elem(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    GnaBoolTensorPrimitive {
        shape: lhs.shape,
        buffer: lhs.buffer,
    }
}

pub(super) fn int_lower_equal(
    lhs: GnaIntTensorPrimitive,
    _rhs: GnaIntTensorPrimitive,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    GnaBoolTensorPrimitive {
        shape: lhs.shape,
        buffer: lhs.buffer,
    }
}

pub(super) fn int_lower_equal_elem(
    lhs: GnaIntTensorPrimitive,
    _rhs: Scalar,
    _out_dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    GnaBoolTensorPrimitive {
        shape: lhs.shape,
        buffer: lhs.buffer,
    }
}
