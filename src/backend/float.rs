use super::GnaBackend;
use crate::device::GnaDevice;
use crate::tensor::{
    GnaBoolTensorPrimitive, GnaBuffer, GnaFloatTensorPrimitive, GnaIntTensorPrimitive,
};
use burn_backend::ops::FloatTensorOps;
use burn_backend::tensor::IndexingUpdateOp;
use burn_backend::{
    BoolStore, DType, Distribution, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice,
    TensorData,
};

// ---------------------------------------------------------------------------
// FloatTensorOps
// ---------------------------------------------------------------------------
impl FloatTensorOps<Self> for GnaBackend {
    fn float_from_data(data: TensorData, _device: &GnaDevice) -> GnaFloatTensorPrimitive {
        let shape = data.shape().clone();
        let dtype = match data.dtype() {
            DType::F64 => FloatDType::F64,
            DType::F32 => FloatDType::F32,
            DType::F16 => FloatDType::F16,
            DType::BF16 => FloatDType::BF16,
            _ => FloatDType::F32,
        };
        GnaFloatTensorPrimitive {
            shape,
            dtype,
            buffer: GnaBuffer::Bytes(std::sync::Arc::new(data.into_bytes().to_vec())),
        }
    }

    fn float_random(
        _shape: Shape,
        _distribution: Distribution,
        _device: &GnaDevice,
        _dtype: FloatDType,
    ) -> GnaFloatTensorPrimitive {
        unimplemented!("float_random is not supported on GNA hardware (inference-only accelerator)")
    }

    async fn float_into_data(
        tensor: GnaFloatTensorPrimitive,
    ) -> Result<TensorData, ExecutionError> {
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

    fn float_to_device(
        tensor: GnaFloatTensorPrimitive,
        _device: &GnaDevice,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_into_int(
        tensor: GnaFloatTensorPrimitive,
        out_dtype: IntDType,
    ) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive {
            shape: tensor.shape,
            dtype: out_dtype,
            buffer: tensor.buffer,
        }
    }

    fn float_empty(
        shape: Shape,
        _device: &GnaDevice,
        dtype: FloatDType,
    ) -> GnaFloatTensorPrimitive {
        let num_elements: usize = shape.num_elements();
        let bytes_per_elem = DType::from(dtype).size();
        let buf = vec![0u8; num_elements * bytes_per_elem];
        GnaFloatTensorPrimitive {
            shape,
            dtype,
            buffer: GnaBuffer::Bytes(std::sync::Arc::new(buf)),
        }
    }

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

    fn float_swap_dims(
        tensor: GnaFloatTensorPrimitive,
        _dim1: usize,
        _dim2: usize,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_permute(tensor: GnaFloatTensorPrimitive, _axes: &[usize]) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_flip(tensor: GnaFloatTensorPrimitive, _axes: &[usize]) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_reshape(tensor: GnaFloatTensorPrimitive, shape: Shape) -> GnaFloatTensorPrimitive {
        GnaFloatTensorPrimitive { shape, ..tensor }
    }

    fn float_gather(
        _dim: usize,
        tensor: GnaFloatTensorPrimitive,
        _indices: GnaIntTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_scatter(
        _dim: usize,
        tensor: GnaFloatTensorPrimitive,
        _indices: GnaIntTensorPrimitive,
        _value: GnaFloatTensorPrimitive,
        _update: IndexingUpdateOp,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_select(
        tensor: GnaFloatTensorPrimitive,
        _dim: usize,
        _indices: GnaIntTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_select_assign(
        tensor: GnaFloatTensorPrimitive,
        _dim: usize,
        _indices: GnaIntTensorPrimitive,
        _value: GnaFloatTensorPrimitive,
        _update: IndexingUpdateOp,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_slice(tensor: GnaFloatTensorPrimitive, _slices: &[Slice]) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_slice_assign(
        tensor: GnaFloatTensorPrimitive,
        _slices: &[Slice],
        _value: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_mask_where(
        tensor: GnaFloatTensorPrimitive,
        _mask: GnaBoolTensorPrimitive,
        _value: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_mask_fill(
        tensor: GnaFloatTensorPrimitive,
        _mask: GnaBoolTensorPrimitive,
        _value: Scalar,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

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

    fn float_sum(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_sum_dim(tensor: GnaFloatTensorPrimitive, _dim: usize) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_mean_dim(tensor: GnaFloatTensorPrimitive, _dim: usize) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_cumsum(tensor: GnaFloatTensorPrimitive, _dim: usize) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_cumprod(tensor: GnaFloatTensorPrimitive, _dim: usize) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_cummin(tensor: GnaFloatTensorPrimitive, _dim: usize) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_cummax(tensor: GnaFloatTensorPrimitive, _dim: usize) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_cast(tensor: GnaFloatTensorPrimitive, dtype: FloatDType) -> GnaFloatTensorPrimitive {
        GnaFloatTensorPrimitive { dtype, ..tensor }
    }

    fn float_exp(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_log(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_log1p(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
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

    fn float_sqrt(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_abs(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_cos(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_sin(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_tan(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_cosh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_sinh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_tanh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_acos(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_acosh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_asin(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_asinh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_atan(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_atanh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_atan2(
        lhs: GnaFloatTensorPrimitive,
        _rhs: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_round(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_floor(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_ceil(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_trunc(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_erf(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_argmax(
        tensor: GnaFloatTensorPrimitive,
        _dim: usize,
        out_dtype: IntDType,
    ) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive {
            shape: tensor.shape,
            dtype: out_dtype,
            buffer: tensor.buffer,
        }
    }

    fn float_argtopk(
        tensor: GnaFloatTensorPrimitive,
        _dim: usize,
        _k: usize,
        out_dtype: IntDType,
    ) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive {
            shape: tensor.shape,
            dtype: out_dtype,
            buffer: tensor.buffer,
        }
    }

    fn float_argmin(
        tensor: GnaFloatTensorPrimitive,
        _dim: usize,
        out_dtype: IntDType,
    ) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive {
            shape: tensor.shape,
            dtype: out_dtype,
            buffer: tensor.buffer,
        }
    }

    fn float_expand(tensor: GnaFloatTensorPrimitive, shape: Shape) -> GnaFloatTensorPrimitive {
        GnaFloatTensorPrimitive { shape, ..tensor }
    }

    fn float_unfold(
        tensor: GnaFloatTensorPrimitive,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }
}
