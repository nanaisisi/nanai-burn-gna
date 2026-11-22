use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::buffer::GnaBuffer;
use burn_backend::{Scalar, TensorMetadata};

pub(super) fn bool_equal(
    lhs: GnaBoolTensorPrimitive,
    rhs: GnaBoolTensorPrimitive,
) -> GnaBoolTensorPrimitive {
    let out: Vec<u8> = lhs
        .buffer()
        .as_slice()
        .iter()
        .zip(rhs.buffer().as_slice().iter())
        .map(|(&a, &b)| u8::from((a != 0) == (b != 0)))
        .collect();
    GnaBoolTensorPrimitive::new(lhs.shape(), GnaBuffer::from_bytes(out))
}

pub(super) fn bool_equal_elem(lhs: GnaBoolTensorPrimitive, rhs: Scalar) -> GnaBoolTensorPrimitive {
    let r = rhs.elem::<bool>();
    let out: Vec<u8> = lhs
        .buffer()
        .as_slice()
        .iter()
        .map(|&a| u8::from((a != 0) == r))
        .collect();
    GnaBoolTensorPrimitive::new(lhs.shape(), GnaBuffer::from_bytes(out))
}

pub(super) fn bool_not(tensor: GnaBoolTensorPrimitive) -> GnaBoolTensorPrimitive {
    let out: Vec<u8> = tensor
        .buffer()
        .as_slice()
        .iter()
        .map(|&a| u8::from(a == 0))
        .collect();
    GnaBoolTensorPrimitive::new(tensor.shape(), GnaBuffer::from_bytes(out))
}

pub(super) fn bool_and(
    lhs: GnaBoolTensorPrimitive,
    rhs: GnaBoolTensorPrimitive,
) -> GnaBoolTensorPrimitive {
    let out: Vec<u8> = lhs
        .buffer()
        .as_slice()
        .iter()
        .zip(rhs.buffer().as_slice().iter())
        .map(|(&a, &b)| u8::from(a != 0 && b != 0))
        .collect();
    GnaBoolTensorPrimitive::new(lhs.shape(), GnaBuffer::from_bytes(out))
}

pub(super) fn bool_or(
    lhs: GnaBoolTensorPrimitive,
    rhs: GnaBoolTensorPrimitive,
) -> GnaBoolTensorPrimitive {
    let out: Vec<u8> = lhs
        .buffer()
        .as_slice()
        .iter()
        .zip(rhs.buffer().as_slice().iter())
        .map(|(&a, &b)| u8::from(a != 0 || b != 0))
        .collect();
    GnaBoolTensorPrimitive::new(lhs.shape(), GnaBuffer::from_bytes(out))
}
