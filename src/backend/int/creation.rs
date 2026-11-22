use crate::device::GnaDevice;
use crate::tensor::{GnaBuffer, GnaFloatTensorPrimitive, GnaIntTensorPrimitive};
use burn_backend::{DType, Distribution, ExecutionError, FloatDType, IntDType, Shape, TensorData};

pub(super) fn int_empty(
    shape: Shape,
    _device: &GnaDevice,
    dtype: IntDType,
) -> GnaIntTensorPrimitive {
    let num = shape.num_elements();
    let bytes_per_elem = DType::from(dtype).size();
    GnaIntTensorPrimitive {
        shape,
        dtype,
        buffer: GnaBuffer::zeros(num * bytes_per_elem),
    }
}

pub(super) async fn int_into_data(
    tensor: GnaIntTensorPrimitive,
) -> Result<TensorData, ExecutionError> {
    let data =
        TensorData::from_bytes_vec(tensor.buffer.to_vec(), tensor.shape, tensor.dtype.into());
    Ok(data)
}

pub(super) fn int_from_data(data: TensorData, _device: &GnaDevice) -> GnaIntTensorPrimitive {
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
        buffer: GnaBuffer::from_bytes(data.into_bytes().to_vec()),
    }
}

pub(super) fn int_to_device(
    tensor: GnaIntTensorPrimitive,
    _device: &GnaDevice,
) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_reshape(tensor: GnaIntTensorPrimitive, shape: Shape) -> GnaIntTensorPrimitive {
    GnaIntTensorPrimitive { shape, ..tensor }
}

pub(super) fn int_into_float(
    tensor: GnaIntTensorPrimitive,
    out_dtype: FloatDType,
) -> GnaFloatTensorPrimitive {
    GnaFloatTensorPrimitive {
        shape: tensor.shape,
        dtype: out_dtype,
        buffer: tensor.buffer,
    }
}

pub(super) fn int_random(
    _shape: Shape,
    _distribution: Distribution,
    _device: &GnaDevice,
    _dtype: IntDType,
) -> GnaIntTensorPrimitive {
    unimplemented!("int_random is not supported on GNA hardware")
}

pub(super) fn int_expand(tensor: GnaIntTensorPrimitive, shape: Shape) -> GnaIntTensorPrimitive {
    GnaIntTensorPrimitive { shape, ..tensor }
}

pub(super) fn int_cast(tensor: GnaIntTensorPrimitive, dtype: IntDType) -> GnaIntTensorPrimitive {
    GnaIntTensorPrimitive { dtype, ..tensor }
}
