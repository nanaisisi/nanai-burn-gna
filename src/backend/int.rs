use super::GnaBackend;
use crate::device::GnaDevice;
use crate::tensor::{
    GnaBoolTensorPrimitive, GnaBuffer, GnaFloatTensorPrimitive, GnaIntTensorPrimitive,
};
use burn_backend::ops::IntTensorOps;
use burn_backend::tensor::IndexingUpdateOp;
use burn_backend::{
    BoolStore, DType, Distribution, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice,
    TensorData,
};

// ---------------------------------------------------------------------------
// IntTensorOps
// ---------------------------------------------------------------------------
impl IntTensorOps<Self> for GnaBackend {
    fn int_empty(shape: Shape, _device: &GnaDevice, dtype: IntDType) -> GnaIntTensorPrimitive {
        let num = shape.num_elements();
        let bytes_per_elem = DType::from(dtype).size();
        GnaIntTensorPrimitive {
            shape,
            dtype,
            buffer: GnaBuffer::Bytes(std::sync::Arc::new(vec![0u8; num * bytes_per_elem])),
        }
    }

    async fn int_into_data(tensor: GnaIntTensorPrimitive) -> Result<TensorData, ExecutionError> {
        match tensor.buffer {
            GnaBuffer::Bytes(bytes) => {
                let data = TensorData::from_bytes_vec(
                    bytes.as_slice().to_vec(),
                    tensor.shape,
                    tensor.dtype.into(),
                );
                Ok(data)
            }
        }
    }

    fn int_from_data(data: TensorData, _device: &GnaDevice) -> GnaIntTensorPrimitive {
        let shape = data.shape().clone();
        let dtype = match data.dtype() {
            DType::I64 => IntDType::I64,
            DType::I32 => IntDType::I32,
            DType::I16 => IntDType::I16,
            DType::I8 => IntDType::I8,
            _ => IntDType::I32,
        };
        GnaIntTensorPrimitive {
            shape,
            dtype,
            buffer: GnaBuffer::Bytes(std::sync::Arc::new(data.into_bytes().to_vec())),
        }
    }

    fn int_to_device(tensor: GnaIntTensorPrimitive, _device: &GnaDevice) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_reshape(tensor: GnaIntTensorPrimitive, shape: Shape) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive { shape, ..tensor }
    }

    fn int_slice(tensor: GnaIntTensorPrimitive, _slices: &[Slice]) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_slice_assign(
        tensor: GnaIntTensorPrimitive,
        _slices: &[Slice],
        _value: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_into_float(
        tensor: GnaIntTensorPrimitive,
        out_dtype: FloatDType,
    ) -> GnaFloatTensorPrimitive {
        GnaFloatTensorPrimitive {
            shape: tensor.shape,
            dtype: out_dtype,
            buffer: tensor.buffer,
        }
    }

    fn int_mask_where(
        tensor: GnaIntTensorPrimitive,
        _mask: GnaBoolTensorPrimitive,
        _value: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_mask_fill(
        tensor: GnaIntTensorPrimitive,
        _mask: GnaBoolTensorPrimitive,
        _value: Scalar,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_gather(
        _dim: usize,
        tensor: GnaIntTensorPrimitive,
        _indices: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_scatter(
        _dim: usize,
        tensor: GnaIntTensorPrimitive,
        _indices: GnaIntTensorPrimitive,
        _value: GnaIntTensorPrimitive,
        _update: IndexingUpdateOp,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_select(
        tensor: GnaIntTensorPrimitive,
        _dim: usize,
        _indices: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_select_assign(
        tensor: GnaIntTensorPrimitive,
        _dim: usize,
        _indices: GnaIntTensorPrimitive,
        _value: GnaIntTensorPrimitive,
        _update: IndexingUpdateOp,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_equal(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_equal_elem(
        lhs: GnaIntTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_greater(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_greater_elem(
        lhs: GnaIntTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_greater_equal(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_greater_equal_elem(
        lhs: GnaIntTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_lower(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_lower_elem(
        lhs: GnaIntTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_lower_equal(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_lower_equal_elem(
        lhs: GnaIntTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_add(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_add_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_sub(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_sub_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_mul(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_mul_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_div(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
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

    fn int_argtopk(tensor: GnaIntTensorPrimitive, _dim: usize, _k: usize) -> GnaIntTensorPrimitive {
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

    fn int_swap_dims(
        tensor: GnaIntTensorPrimitive,
        _dim1: usize,
        _dim2: usize,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_permute(tensor: GnaIntTensorPrimitive, _axes: &[usize]) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_flip(tensor: GnaIntTensorPrimitive, _axes: &[usize]) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_random(
        _shape: Shape,
        _distribution: Distribution,
        _device: &GnaDevice,
        _dtype: IntDType,
    ) -> GnaIntTensorPrimitive {
        unimplemented!("int_random is not supported on GNA hardware")
    }

    fn int_expand(tensor: GnaIntTensorPrimitive, shape: Shape) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive { shape, ..tensor }
    }

    fn bitwise_and(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_and_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_or(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_or_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_xor(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_xor_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_not(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        tensor
    }

    fn bitwise_left_shift(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_left_shift_scalar(
        lhs: GnaIntTensorPrimitive,
        _rhs: Scalar,
    ) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_right_shift(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_right_shift_scalar(
        lhs: GnaIntTensorPrimitive,
        _rhs: Scalar,
    ) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_cast(tensor: GnaIntTensorPrimitive, dtype: IntDType) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive { dtype, ..tensor }
    }

    fn int_unfold(
        tensor: GnaIntTensorPrimitive,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> GnaIntTensorPrimitive {
        tensor
    }
}
