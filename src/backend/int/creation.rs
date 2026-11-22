use crate::device::GnaDevice;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::{DType, Distribution, ExecutionError, FloatDType, IntDType, Shape, TensorData};

pub(super) fn int_empty(
    shape: Shape,
    _device: &GnaDevice,
    dtype: IntDType,
) -> GnaIntTensorPrimitive {
    let num = shape.num_elements();
    let bytes_per_elem = DType::from(dtype).size();
    GnaIntTensorPrimitive::new(shape, dtype, GnaBuffer::zeros(num * bytes_per_elem))
}

pub(super) async fn int_into_data(
    tensor: GnaIntTensorPrimitive,
) -> Result<TensorData, ExecutionError> {
    let (shape, dtype, buffer) = tensor.into_parts();
    let data = TensorData::from_bytes_vec(buffer.into_vec(), shape, dtype.into());
    Ok(data)
}

pub(super) fn int_from_data(data: TensorData, _device: &GnaDevice) -> GnaIntTensorPrimitive {
    let shape = data.shape().clone();
    let dtype = match data.dtype() {
        DType::I64 => IntDType::I64,
        DType::I32 => IntDType::I32,
        DType::I16 => IntDType::I16,
        DType::I8 => IntDType::I8,
        DType::U64 => IntDType::U64,
        DType::U32 => IntDType::U32,
        DType::U16 => IntDType::U16,
        DType::U8 => IntDType::U8,
        _ => IntDType::I32,
    };
    GnaIntTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(data.into_bytes().to_vec()),
    )
}

pub(super) fn int_to_device(
    tensor: GnaIntTensorPrimitive,
    _device: &GnaDevice,
) -> GnaIntTensorPrimitive {
    tensor
}

pub(super) fn int_reshape(tensor: GnaIntTensorPrimitive, shape: Shape) -> GnaIntTensorPrimitive {
    tensor.with_shape(shape)
}

pub(super) fn int_into_float(
    tensor: GnaIntTensorPrimitive,
    out_dtype: FloatDType,
) -> GnaFloatTensorPrimitive {
    crate::backend::gna_exec::layout::convert_int_to_float(tensor, out_dtype)
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
    use burn_backend::TensorMetadata;
    let elem_size = DType::from(tensor.dtype()).size();
    let out_bytes = crate::backend::gna_exec::layout::expand_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        &shape,
        elem_size,
    );
    GnaIntTensorPrimitive::new(shape, tensor.dtype(), GnaBuffer::from_bytes(out_bytes))
}

pub(super) fn int_cast(tensor: GnaIntTensorPrimitive, dtype: IntDType) -> GnaIntTensorPrimitive {
    tensor.with_dtype(dtype)
}
