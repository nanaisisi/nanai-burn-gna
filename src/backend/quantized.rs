use super::GnaBackend;
use crate::device::GnaDevice;
use crate::tensor::{
    GnaBuffer, GnaFloatTensorPrimitive, GnaIntTensorPrimitive, GnaQTensorPrimitive,
};
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
            _ => GnaQTensorPrimitive::default_scheme(),
        };
        GnaQTensorPrimitive {
            shape: data.shape().clone(),
            scheme,
            buffer: GnaBuffer::from_bytes(data.into_bytes().to_vec()),
        }
    }

    fn quantize(
        tensor: GnaFloatTensorPrimitive,
        scheme: &QuantScheme,
        _qparams: QuantizationParametersPrimitive<Self>,
    ) -> GnaQTensorPrimitive {
        GnaQTensorPrimitive {
            shape: tensor.shape,
            scheme: *scheme,
            buffer: tensor.buffer,
        }
    }

    fn dequantize(tensor: GnaQTensorPrimitive, dtype: FloatDType) -> GnaFloatTensorPrimitive {
        GnaFloatTensorPrimitive {
            shape: tensor.shape,
            dtype,
            buffer: tensor.buffer,
        }
    }

    fn q_to_device(tensor: GnaQTensorPrimitive, _device: &GnaDevice) -> GnaQTensorPrimitive {
        tensor
    }

    fn q_reshape(tensor: GnaQTensorPrimitive, shape: Shape) -> GnaQTensorPrimitive {
        GnaQTensorPrimitive { shape, ..tensor }
    }

    async fn q_into_data(tensor: GnaQTensorPrimitive) -> Result<TensorData, ExecutionError> {
        let data = TensorData::from_bytes_vec(
            tensor.buffer.to_vec(),
            tensor.shape,
            DType::QFloat(tensor.scheme),
        );
        Ok(data)
    }

    fn q_expand(tensor: GnaQTensorPrimitive, shape: Shape) -> GnaQTensorPrimitive {
        GnaQTensorPrimitive { shape, ..tensor }
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
