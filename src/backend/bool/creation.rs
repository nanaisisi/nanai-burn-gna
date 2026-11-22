use crate::backend::gna_exec::{encode_floats, encode_ints, layout};
use crate::device::GnaDevice;
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::{
    BoolStore, DType, ExecutionError, FloatDType, IntDType, Shape, TensorData, TensorMetadata,
};

pub(super) fn bool_empty(
    shape: Shape,
    _device: &GnaDevice,
    _dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    let num = shape.num_elements();
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::zeros(num))
}

pub(super) fn bool_zeros(
    shape: Shape,
    device: &GnaDevice,
    dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    bool_empty(shape, device, dtype)
}

pub(super) fn bool_ones(
    shape: Shape,
    _device: &GnaDevice,
    _dtype: BoolStore,
) -> GnaBoolTensorPrimitive {
    let num = shape.num_elements();
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::filled(1, num))
}

pub(super) async fn bool_into_data(
    tensor: GnaBoolTensorPrimitive,
) -> Result<TensorData, ExecutionError> {
    let (shape, buffer) = tensor.into_parts();
    let data = TensorData::from_bytes_vec(buffer.into_vec(), shape, DType::Bool(BoolStore::Native));
    Ok(data)
}

pub(super) fn bool_from_data(data: TensorData, _device: &GnaDevice) -> GnaBoolTensorPrimitive {
    GnaBoolTensorPrimitive::new(
        data.shape().clone(),
        GnaBuffer::from_bytes(data.into_bytes().to_vec()),
    )
}

pub(super) fn bool_into_int(
    tensor: GnaBoolTensorPrimitive,
    out_dtype: IntDType,
) -> GnaIntTensorPrimitive {
    let (shape, buffer) = tensor.into_parts();
    let vals: Vec<i64> = buffer
        .as_slice()
        .iter()
        .map(|&b| i64::from(b != 0))
        .collect();
    GnaIntTensorPrimitive::new(
        shape,
        out_dtype,
        GnaBuffer::from_bytes(encode_ints(&vals, out_dtype)),
    )
}

pub(super) fn bool_into_float(
    tensor: GnaBoolTensorPrimitive,
    out_dtype: FloatDType,
) -> GnaFloatTensorPrimitive {
    let (shape, buffer) = tensor.into_parts();
    let vals: Vec<f32> = buffer
        .as_slice()
        .iter()
        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
        .collect();
    GnaFloatTensorPrimitive::new(
        shape,
        out_dtype,
        GnaBuffer::from_bytes(encode_floats(&vals, out_dtype)),
    )
}

pub(super) fn bool_to_device(
    tensor: GnaBoolTensorPrimitive,
    _device: &GnaDevice,
) -> GnaBoolTensorPrimitive {
    tensor
}

pub(super) fn bool_reshape(tensor: GnaBoolTensorPrimitive, shape: Shape) -> GnaBoolTensorPrimitive {
    tensor.with_shape(shape)
}

pub(super) fn bool_expand(tensor: GnaBoolTensorPrimitive, shape: Shape) -> GnaBoolTensorPrimitive {
    let bytes = layout::expand_bytes(
        layout::shape::ByteTensorView::new(tensor.buffer().as_slice(), &tensor.shape(), 1),
        &shape,
    );
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
}
