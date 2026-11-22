use super::GnaBackend;
use crate::device::GnaDevice;
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::ops::BoolTensorOps;
use burn_backend::{
    BoolStore, DType, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
};

// ---------------------------------------------------------------------------
// BoolTensorOps
// ---------------------------------------------------------------------------
impl BoolTensorOps<Self> for GnaBackend {
    fn bool_empty(shape: Shape, _device: &GnaDevice, _dtype: BoolStore) -> GnaBoolTensorPrimitive {
        let num = shape.num_elements();
        GnaBoolTensorPrimitive::new(shape, GnaBuffer::zeros(num))
    }

    fn bool_zeros(shape: Shape, device: &GnaDevice, dtype: BoolStore) -> GnaBoolTensorPrimitive {
        Self::bool_empty(shape, device, dtype)
    }

    fn bool_ones(shape: Shape, _device: &GnaDevice, _dtype: BoolStore) -> GnaBoolTensorPrimitive {
        let num = shape.num_elements();
        GnaBoolTensorPrimitive::new(shape, GnaBuffer::filled(1, num))
    }

    async fn bool_into_data(tensor: GnaBoolTensorPrimitive) -> Result<TensorData, ExecutionError> {
        let (shape, buffer) = tensor.into_parts();
        let data =
            TensorData::from_bytes_vec(buffer.into_vec(), shape, DType::Bool(BoolStore::Native));
        Ok(data)
    }

    fn bool_from_data(data: TensorData, _device: &GnaDevice) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive::new(
            data.shape().clone(),
            GnaBuffer::from_bytes(data.into_bytes().to_vec()),
        )
    }

    fn bool_into_int(tensor: GnaBoolTensorPrimitive, out_dtype: IntDType) -> GnaIntTensorPrimitive {
        let (shape, buffer) = tensor.into_parts();
        GnaIntTensorPrimitive::new(shape, out_dtype, buffer)
    }

    fn bool_into_float(
        tensor: GnaBoolTensorPrimitive,
        out_dtype: FloatDType,
    ) -> GnaFloatTensorPrimitive {
        let (shape, buffer) = tensor.into_parts();
        GnaFloatTensorPrimitive::new(shape, out_dtype, buffer)
    }

    fn bool_to_device(
        tensor: GnaBoolTensorPrimitive,
        _device: &GnaDevice,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_reshape(tensor: GnaBoolTensorPrimitive, shape: Shape) -> GnaBoolTensorPrimitive {
        tensor.with_shape(shape)
    }

    fn bool_slice(tensor: GnaBoolTensorPrimitive, _slices: &[Slice]) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_slice_assign(
        tensor: GnaBoolTensorPrimitive,
        _slices: &[Slice],
        _value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_mask_where(
        tensor: GnaBoolTensorPrimitive,
        _mask: GnaBoolTensorPrimitive,
        _value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_mask_fill(
        tensor: GnaBoolTensorPrimitive,
        _mask: GnaBoolTensorPrimitive,
        _value: Scalar,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_gather(
        _dim: usize,
        tensor: GnaBoolTensorPrimitive,
        _indices: GnaIntTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_scatter_or(
        _dim: usize,
        tensor: GnaBoolTensorPrimitive,
        _indices: GnaIntTensorPrimitive,
        _value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_select(
        tensor: GnaBoolTensorPrimitive,
        _dim: usize,
        _indices: GnaIntTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_select_or(
        tensor: GnaBoolTensorPrimitive,
        _dim: usize,
        _indices: GnaIntTensorPrimitive,
        _value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_equal(
        lhs: GnaBoolTensorPrimitive,
        _rhs: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        lhs
    }

    fn bool_equal_elem(lhs: GnaBoolTensorPrimitive, _rhs: Scalar) -> GnaBoolTensorPrimitive {
        lhs
    }

    fn bool_not(tensor: GnaBoolTensorPrimitive) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_and(
        lhs: GnaBoolTensorPrimitive,
        _rhs: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        lhs
    }

    fn bool_or(
        lhs: GnaBoolTensorPrimitive,
        _rhs: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        lhs
    }

    fn bool_swap_dims(
        tensor: GnaBoolTensorPrimitive,
        _dim1: usize,
        _dim2: usize,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_permute(tensor: GnaBoolTensorPrimitive, _axes: &[usize]) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_flip(tensor: GnaBoolTensorPrimitive, _axes: &[usize]) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_expand(tensor: GnaBoolTensorPrimitive, shape: Shape) -> GnaBoolTensorPrimitive {
        tensor.with_shape(shape)
    }

    fn bool_unfold(
        tensor: GnaBoolTensorPrimitive,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }
}
