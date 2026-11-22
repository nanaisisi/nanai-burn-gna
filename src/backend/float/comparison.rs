macro_rules! comparison_ops {
    () => {
        fn float_equal(
            lhs: GnaFloatTensorPrimitive,
            _rhs: GnaFloatTensorPrimitive,
            _out_dtype: BoolStore,
        ) -> GnaBoolTensorPrimitive {
            GnaBoolTensorPrimitive {
                shape: lhs.shape,
                buffer: lhs.buffer,
            }
        }

        fn float_equal_elem(
            lhs: GnaFloatTensorPrimitive,
            _rhs: Scalar,
            _out_dtype: BoolStore,
        ) -> GnaBoolTensorPrimitive {
            GnaBoolTensorPrimitive {
                shape: lhs.shape,
                buffer: lhs.buffer,
            }
        }

        fn float_greater(
            lhs: GnaFloatTensorPrimitive,
            _rhs: GnaFloatTensorPrimitive,
            _out_dtype: BoolStore,
        ) -> GnaBoolTensorPrimitive {
            GnaBoolTensorPrimitive {
                shape: lhs.shape,
                buffer: lhs.buffer,
            }
        }

        fn float_greater_elem(
            lhs: GnaFloatTensorPrimitive,
            _rhs: Scalar,
            _out_dtype: BoolStore,
        ) -> GnaBoolTensorPrimitive {
            GnaBoolTensorPrimitive {
                shape: lhs.shape,
                buffer: lhs.buffer,
            }
        }

        fn float_greater_equal(
            lhs: GnaFloatTensorPrimitive,
            _rhs: GnaFloatTensorPrimitive,
            _out_dtype: BoolStore,
        ) -> GnaBoolTensorPrimitive {
            GnaBoolTensorPrimitive {
                shape: lhs.shape,
                buffer: lhs.buffer,
            }
        }

        fn float_greater_equal_elem(
            lhs: GnaFloatTensorPrimitive,
            _rhs: Scalar,
            _out_dtype: BoolStore,
        ) -> GnaBoolTensorPrimitive {
            GnaBoolTensorPrimitive {
                shape: lhs.shape,
                buffer: lhs.buffer,
            }
        }

        fn float_lower(
            lhs: GnaFloatTensorPrimitive,
            _rhs: GnaFloatTensorPrimitive,
            _out_dtype: BoolStore,
        ) -> GnaBoolTensorPrimitive {
            GnaBoolTensorPrimitive {
                shape: lhs.shape,
                buffer: lhs.buffer,
            }
        }

        fn float_lower_elem(
            lhs: GnaFloatTensorPrimitive,
            _rhs: Scalar,
            _out_dtype: BoolStore,
        ) -> GnaBoolTensorPrimitive {
            GnaBoolTensorPrimitive {
                shape: lhs.shape,
                buffer: lhs.buffer,
            }
        }

        fn float_lower_equal(
            lhs: GnaFloatTensorPrimitive,
            _rhs: GnaFloatTensorPrimitive,
            _out_dtype: BoolStore,
        ) -> GnaBoolTensorPrimitive {
            GnaBoolTensorPrimitive {
                shape: lhs.shape,
                buffer: lhs.buffer,
            }
        }

        fn float_lower_equal_elem(
            lhs: GnaFloatTensorPrimitive,
            _rhs: Scalar,
            _out_dtype: BoolStore,
        ) -> GnaBoolTensorPrimitive {
            GnaBoolTensorPrimitive {
                shape: lhs.shape,
                buffer: lhs.buffer,
            }
        }
    };
}

pub(super) use comparison_ops;
