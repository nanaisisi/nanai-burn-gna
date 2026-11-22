use super::GnaBackend;
use crate::device::GnaDevice;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use crate::tensor::quantized::GnaQTensorPrimitive;
use burn_backend::ops::QTensorOps;
use burn_backend::quantization::{QuantScheme, QuantizationParametersPrimitive};
use burn_backend::{ExecutionError, FloatDType, Shape, Slice, TensorData};

mod creation;
mod indexing;

impl QTensorOps<Self> for GnaBackend {
    fn q_from_data(data: TensorData, device: &GnaDevice) -> GnaQTensorPrimitive {
        creation::q_from_data(data, device)
    }

    fn quantize(
        tensor: GnaFloatTensorPrimitive,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> GnaQTensorPrimitive {
        creation::quantize(tensor, scheme, qparams)
    }

    fn dequantize(tensor: GnaQTensorPrimitive, dtype: FloatDType) -> GnaFloatTensorPrimitive {
        creation::dequantize(tensor, dtype)
    }

    fn q_to_device(tensor: GnaQTensorPrimitive, device: &GnaDevice) -> GnaQTensorPrimitive {
        creation::q_to_device(tensor, device)
    }

    fn q_reshape(tensor: GnaQTensorPrimitive, shape: Shape) -> GnaQTensorPrimitive {
        creation::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: GnaQTensorPrimitive) -> Result<TensorData, ExecutionError> {
        creation::q_into_data(tensor).await
    }

    fn q_expand(tensor: GnaQTensorPrimitive, shape: Shape) -> GnaQTensorPrimitive {
        creation::q_expand(tensor, shape)
    }

    fn q_swap_dims(tensor: GnaQTensorPrimitive, dim1: usize, dim2: usize) -> GnaQTensorPrimitive {
        indexing::q_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: GnaQTensorPrimitive, axes: &[usize]) -> GnaQTensorPrimitive {
        indexing::q_permute(tensor, axes)
    }

    fn q_flip(tensor: GnaQTensorPrimitive, axes: &[usize]) -> GnaQTensorPrimitive {
        indexing::q_flip(tensor, axes)
    }

    fn q_select(
        tensor: GnaQTensorPrimitive,
        dim: usize,
        indices: GnaIntTensorPrimitive,
    ) -> GnaQTensorPrimitive {
        indexing::q_select(tensor, dim, indices)
    }

    fn q_slice(tensor: GnaQTensorPrimitive, slices: &[Slice]) -> GnaQTensorPrimitive {
        indexing::q_slice(tensor, slices)
    }
}
