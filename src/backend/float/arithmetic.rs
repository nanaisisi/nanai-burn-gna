macro_rules! arithmetic_ops {
    () => {
        fn float_add(
            lhs: GnaFloatTensorPrimitive,
            _rhs: GnaFloatTensorPrimitive,
        ) -> GnaFloatTensorPrimitive {
            lhs
        }

        fn float_add_scalar(lhs: GnaFloatTensorPrimitive, _rhs: Scalar) -> GnaFloatTensorPrimitive {
            lhs
        }

        fn float_sub(
            lhs: GnaFloatTensorPrimitive,
            _rhs: GnaFloatTensorPrimitive,
        ) -> GnaFloatTensorPrimitive {
            lhs
        }

        fn float_sub_scalar(lhs: GnaFloatTensorPrimitive, _rhs: Scalar) -> GnaFloatTensorPrimitive {
            lhs
        }

        fn float_mul(
            lhs: GnaFloatTensorPrimitive,
            _rhs: GnaFloatTensorPrimitive,
        ) -> GnaFloatTensorPrimitive {
            lhs
        }

        fn float_mul_scalar(lhs: GnaFloatTensorPrimitive, _rhs: Scalar) -> GnaFloatTensorPrimitive {
            lhs
        }

        fn float_div(
            lhs: GnaFloatTensorPrimitive,
            _rhs: GnaFloatTensorPrimitive,
        ) -> GnaFloatTensorPrimitive {
            lhs
        }

        fn float_div_scalar(lhs: GnaFloatTensorPrimitive, _rhs: Scalar) -> GnaFloatTensorPrimitive {
            lhs
        }

        fn float_remainder(
            lhs: GnaFloatTensorPrimitive,
            _rhs: GnaFloatTensorPrimitive,
        ) -> GnaFloatTensorPrimitive {
            lhs
        }

        fn float_remainder_scalar(
            lhs: GnaFloatTensorPrimitive,
            _rhs: Scalar,
        ) -> GnaFloatTensorPrimitive {
            lhs
        }

        fn float_matmul(
            lhs: GnaFloatTensorPrimitive,
            _rhs: GnaFloatTensorPrimitive,
        ) -> GnaFloatTensorPrimitive {
            lhs
        }

        fn float_cross(
            lhs: GnaFloatTensorPrimitive,
            _rhs: GnaFloatTensorPrimitive,
            _dim: usize,
        ) -> GnaFloatTensorPrimitive {
            lhs
        }

        fn float_recip(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
            tensor
        }

        fn float_powf(
            lhs: GnaFloatTensorPrimitive,
            _rhs: GnaFloatTensorPrimitive,
        ) -> GnaFloatTensorPrimitive {
            lhs
        }

        fn float_powf_scalar_impl(
            tensor: GnaFloatTensorPrimitive,
            _value: Scalar,
        ) -> GnaFloatTensorPrimitive {
            tensor
        }

        fn float_atan2(
            lhs: GnaFloatTensorPrimitive,
            _rhs: GnaFloatTensorPrimitive,
        ) -> GnaFloatTensorPrimitive {
            lhs
        }
    };
}

pub(super) use arithmetic_ops;
