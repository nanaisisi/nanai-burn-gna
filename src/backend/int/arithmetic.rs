macro_rules! arithmetic_ops {
    () => {
        fn int_add(
            lhs: GnaIntTensorPrimitive,
            _rhs: GnaIntTensorPrimitive,
        ) -> GnaIntTensorPrimitive {
            lhs
        }

        fn int_add_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
            lhs
        }

        fn int_sub(
            lhs: GnaIntTensorPrimitive,
            _rhs: GnaIntTensorPrimitive,
        ) -> GnaIntTensorPrimitive {
            lhs
        }

        fn int_sub_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
            lhs
        }

        fn int_mul(
            lhs: GnaIntTensorPrimitive,
            _rhs: GnaIntTensorPrimitive,
        ) -> GnaIntTensorPrimitive {
            lhs
        }

        fn int_mul_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
            lhs
        }

        fn int_div(
            lhs: GnaIntTensorPrimitive,
            _rhs: GnaIntTensorPrimitive,
        ) -> GnaIntTensorPrimitive {
            lhs
        }

        fn int_div_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
            lhs
        }

        fn int_remainder(
            lhs: GnaIntTensorPrimitive,
            _rhs: GnaIntTensorPrimitive,
        ) -> GnaIntTensorPrimitive {
            lhs
        }

        fn int_remainder_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
            lhs
        }

        fn int_matmul(
            lhs: GnaIntTensorPrimitive,
            _rhs: GnaIntTensorPrimitive,
        ) -> GnaIntTensorPrimitive {
            lhs
        }
    };
}

pub(super) use arithmetic_ops;
