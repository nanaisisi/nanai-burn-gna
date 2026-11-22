use super::GnaBackend;
use crate::device::GnaDevice;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use crate::tensor::quantized::GnaQTensorPrimitive;
use burn_backend::ops::QTensorOps;
use burn_backend::quantization::{QuantScheme, QuantizationParametersPrimitive};
use burn_backend::{DType, ExecutionError, FloatDType, Shape, Slice, TensorData};

// ---------------------------------------------------------------------------
// QTensorOps
// ---------------------------------------------------------------------------
impl QTensorOps<Self> for GnaBackend {
    fn q_from_data(data: TensorData, _device: &GnaDevice) -> GnaQTensorPrimitive {
        let scheme = match data.dtype() {
            DType::QFloat(scheme) => scheme,
            _ => GnaDevice::default_quant_scheme(),
        };
        GnaQTensorPrimitive::new(
            data.shape().clone(),
            scheme,
            GnaBuffer::from_bytes(data.into_bytes().to_vec()),
        )
    }

    fn quantize(
        tensor: GnaFloatTensorPrimitive,
        scheme: &QuantScheme,
        _qparams: QuantizationParametersPrimitive<Self>,
    ) -> GnaQTensorPrimitive {
        let (shape, _dtype, buffer) = tensor.into_parts();
        GnaQTensorPrimitive::new(shape, *scheme, buffer)
    }

    fn dequantize(tensor: GnaQTensorPrimitive, dtype: FloatDType) -> GnaFloatTensorPrimitive {
        let (shape, _scheme, buffer) = tensor.into_parts();
        GnaFloatTensorPrimitive::new(shape, dtype, buffer)
    }

    fn q_to_device(tensor: GnaQTensorPrimitive, _device: &GnaDevice) -> GnaQTensorPrimitive {
        tensor
    }

    fn q_reshape(tensor: GnaQTensorPrimitive, shape: Shape) -> GnaQTensorPrimitive {
        tensor.with_shape(shape)
    }

    async fn q_into_data(tensor: GnaQTensorPrimitive) -> Result<TensorData, ExecutionError> {
        let (shape, scheme, buffer) = tensor.into_parts();
        let data = TensorData::from_bytes_vec(buffer.to_vec(), shape, DType::QFloat(scheme));
        Ok(data)
    }

    fn q_expand(tensor: GnaQTensorPrimitive, shape: Shape) -> GnaQTensorPrimitive {
        tensor.with_shape(shape)
    }

    fn q_swap_dims(tensor: GnaQTensorPrimitive, _dim1: usize, _dim2: usize) -> GnaQTensorPrimitive {
        tensor
    }

    fn q_permute(tensor: GnaQTensorPrimitive, _axes: &[usize]) -> GnaQTensorPrimitive {
        tensor
    }

    fn q_flip(tensor: GnaQTensorPrimitive, _axes: &[usize]) -> GnaQTensorPrimitive {
        tensor
    }

    fn q_select(
        tensor: GnaQTensorPrimitive,
        _dim: usize,
        _indices: GnaIntTensorPrimitive,
    ) -> GnaQTensorPrimitive {
        tensor
    }

    fn q_slice(tensor: GnaQTensorPrimitive, _slices: &[Slice]) -> GnaQTensorPrimitive {
        tensor
    }
}
