use crate::backend::GnaBackend;
use crate::backend::gna_exec::module_ops::{
    QLayoutOp, execute_dequantize, execute_quantize, transform_q_bytes,
};
use crate::device::GnaDevice;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::quantized::GnaQTensorPrimitive;
use burn_backend::quantization::{QuantScheme, QuantizationParametersPrimitive};
use burn_backend::{DType, ExecutionError, FloatDType, Shape, TensorData};

pub(super) fn q_from_data(data: TensorData, _device: &GnaDevice) -> GnaQTensorPrimitive {
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

pub(super) fn quantize(
    tensor: GnaFloatTensorPrimitive,
    scheme: &QuantScheme,
    qparams: QuantizationParametersPrimitive<GnaBackend>,
) -> GnaQTensorPrimitive {
    execute_quantize(tensor, scheme, qparams)
}

pub(super) fn dequantize(
    tensor: GnaQTensorPrimitive,
    dtype: FloatDType,
) -> GnaFloatTensorPrimitive {
    execute_dequantize(tensor, dtype)
}

pub(super) fn q_to_device(tensor: GnaQTensorPrimitive, _device: &GnaDevice) -> GnaQTensorPrimitive {
    tensor
}

pub(super) fn q_reshape(tensor: GnaQTensorPrimitive, shape: Shape) -> GnaQTensorPrimitive {
    tensor.with_shape(shape)
}

pub(super) async fn q_into_data(tensor: GnaQTensorPrimitive) -> Result<TensorData, ExecutionError> {
    let (shape, scheme, buffer) = tensor.into_parts();
    let data = TensorData::from_bytes_vec(buffer.into_vec(), shape, DType::QFloat(scheme));
    Ok(data)
}

pub(super) fn q_expand(tensor: GnaQTensorPrimitive, shape: Shape) -> GnaQTensorPrimitive {
    transform_q_bytes(tensor, QLayoutOp::Expand(shape))
}
