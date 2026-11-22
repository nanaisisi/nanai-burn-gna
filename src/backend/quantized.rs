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
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> GnaQTensorPrimitive {
        super::gna_exec::module_ops::execute_quantize(tensor, scheme, qparams)
    }

    fn dequantize(tensor: GnaQTensorPrimitive, dtype: FloatDType) -> GnaFloatTensorPrimitive {
        super::gna_exec::module_ops::execute_dequantize(tensor, dtype)
    }

    fn q_to_device(tensor: GnaQTensorPrimitive, _device: &GnaDevice) -> GnaQTensorPrimitive {
        tensor
    }

    fn q_reshape(tensor: GnaQTensorPrimitive, shape: Shape) -> GnaQTensorPrimitive {
        tensor.with_shape(shape)
    }

    async fn q_into_data(tensor: GnaQTensorPrimitive) -> Result<TensorData, ExecutionError> {
        let (shape, scheme, buffer) = tensor.into_parts();
        let data = TensorData::from_bytes_vec(buffer.into_vec(), shape, DType::QFloat(scheme));
        Ok(data)
    }

    fn q_expand(tensor: GnaQTensorPrimitive, shape: Shape) -> GnaQTensorPrimitive {
        super::gna_exec::module_ops::transform_q_bytes(
            tensor,
            super::gna_exec::module_ops::QLayoutOp::Expand(shape),
        )
    }

    fn q_swap_dims(tensor: GnaQTensorPrimitive, dim1: usize, dim2: usize) -> GnaQTensorPrimitive {
        super::gna_exec::module_ops::transform_q_bytes(
            tensor,
            super::gna_exec::module_ops::QLayoutOp::SwapDims(dim1, dim2),
        )
    }

    fn q_permute(tensor: GnaQTensorPrimitive, axes: &[usize]) -> GnaQTensorPrimitive {
        super::gna_exec::module_ops::transform_q_bytes(
            tensor,
            super::gna_exec::module_ops::QLayoutOp::Permute(axes),
        )
    }

    fn q_flip(tensor: GnaQTensorPrimitive, axes: &[usize]) -> GnaQTensorPrimitive {
        super::gna_exec::module_ops::transform_q_bytes(
            tensor,
            super::gna_exec::module_ops::QLayoutOp::Flip(axes),
        )
    }

    fn q_select(
        tensor: GnaQTensorPrimitive,
        dim: usize,
        indices: GnaIntTensorPrimitive,
    ) -> GnaQTensorPrimitive {
        let idx = super::gna_exec::decode_ints(indices.buffer().as_slice(), indices.dtype());
        super::gna_exec::module_ops::transform_q_bytes(
            tensor,
            super::gna_exec::module_ops::QLayoutOp::Select(dim, &idx),
        )
    }

    fn q_slice(tensor: GnaQTensorPrimitive, slices: &[Slice]) -> GnaQTensorPrimitive {
        super::gna_exec::module_ops::transform_q_bytes(
            tensor,
            super::gna_exec::module_ops::QLayoutOp::Slice(slices),
        )
    }
}
