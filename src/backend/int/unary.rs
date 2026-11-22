macro_rules! unary_ops {
    () => {
        fn int_sum(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
            tensor
        }

        fn int_sum_dim(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
            tensor
        }

        fn int_prod(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
            tensor
        }

        fn int_prod_dim(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
            tensor
        }

        fn int_mean_dim(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
            tensor
        }

        fn int_cumsum(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
            tensor
        }

        fn int_cumprod(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
            tensor
        }

        fn int_cummin(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
            tensor
        }

        fn int_cummax(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
            tensor
        }

        fn int_argmax(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
            GnaIntTensorPrimitive {
                dtype: IntDType::I64,
                ..tensor
            }
        }

        fn int_argtopk(
            tensor: GnaIntTensorPrimitive,
            _dim: usize,
            _k: usize,
        ) -> GnaIntTensorPrimitive {
            GnaIntTensorPrimitive {
                dtype: IntDType::I64,
                ..tensor
            }
        }

        fn int_argmin(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
            GnaIntTensorPrimitive {
                dtype: IntDType::I64,
                ..tensor
            }
        }

        fn int_abs(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
            tensor
        }

        fn int_unfold(
            tensor: GnaIntTensorPrimitive,
            _dim: usize,
            _size: usize,
            _step: usize,
        ) -> GnaIntTensorPrimitive {
            tensor
        }
    };
}

pub(super) use unary_ops;
