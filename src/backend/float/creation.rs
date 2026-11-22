use crate::device::GnaDevice;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::{DType, Distribution, ExecutionError, FloatDType, IntDType, Shape, TensorData};

pub(super) fn float_from_data(data: TensorData, _device: &GnaDevice) -> GnaFloatTensorPrimitive {
    let shape = data.shape().clone();
    let dtype = match data.dtype() {
        DType::F64 => FloatDType::F64,
        DType::F32 => FloatDType::F32,
        DType::F16 => FloatDType::F16,
        DType::BF16 => FloatDType::BF16,
        _ => FloatDType::F32,
    };
    GnaFloatTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(data.into_bytes().to_vec()),
    )
}

pub(super) fn float_random(
    shape: Shape,
    distribution: Distribution,
    _device: &GnaDevice,
    dtype: FloatDType,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::extra::execute_float_random(shape, distribution, dtype)
}

pub(super) async fn float_into_data(
    tensor: GnaFloatTensorPrimitive,
) -> Result<TensorData, ExecutionError> {
    let (shape, dtype, buffer) = tensor.into_parts();
    let data = TensorData::from_bytes_vec(buffer.into_vec(), shape, dtype.into());
    Ok(data)
}

pub(super) fn float_to_device(
    tensor: GnaFloatTensorPrimitive,
    _device: &GnaDevice,
) -> GnaFloatTensorPrimitive {
    tensor
}

pub(super) fn float_into_int(
    tensor: GnaFloatTensorPrimitive,
    out_dtype: IntDType,
) -> GnaIntTensorPrimitive {
    crate::backend::gna_exec::layout::convert_float_to_int(tensor, out_dtype)
}

pub(super) fn float_empty(
    shape: Shape,
    _device: &GnaDevice,
    dtype: FloatDType,
) -> GnaFloatTensorPrimitive {
    let num_elements: usize = shape.num_elements();
    let bytes_per_elem = DType::from(dtype).size();
    GnaFloatTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::zeros(num_elements * bytes_per_elem),
    )
}

pub(super) fn float_cast(
    tensor: GnaFloatTensorPrimitive,
    dtype: FloatDType,
) -> GnaFloatTensorPrimitive {
    use burn_backend::TensorMetadata;
    if tensor.dtype() == dtype {
        return tensor;
    }
    let vals = crate::backend::gna_exec::decode_floats(tensor.buffer().as_slice(), tensor.dtype());
    let bytes = crate::backend::gna_exec::encode_floats(&vals, dtype);
    GnaFloatTensorPrimitive::new(tensor.shape(), dtype, GnaBuffer::from_bytes(bytes))
}

pub(super) fn float_expand(
    tensor: GnaFloatTensorPrimitive,
    shape: Shape,
) -> GnaFloatTensorPrimitive {
    use burn_backend::TensorMetadata;
    let elem_size = DType::from(tensor.dtype()).size();
    let out_bytes = crate::backend::gna_exec::layout::expand_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        &shape,
        elem_size,
    );
    GnaFloatTensorPrimitive::new(shape, tensor.dtype(), GnaBuffer::from_bytes(out_bytes))
}

pub(super) fn float_reshape(
    tensor: GnaFloatTensorPrimitive,
    shape: Shape,
) -> GnaFloatTensorPrimitive {
    tensor.with_shape(shape)
}
