use burn_backend::{Scalar, TensorMetadata};

use crate::backend::gna_exec::{decode_floats, decode_ints, encode_floats, encode_ints, ops};
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;

pub fn scalar_to_le_bytes(scalar: Scalar, elem_size: usize) -> Vec<u8> {
    match elem_size {
        1 => vec![scalar.elem::<u8>()],
        2 => scalar.elem::<i16>().to_le_bytes().to_vec(),
        4 => scalar.elem::<f32>().to_le_bytes().to_vec(),
        8 => scalar.elem::<f64>().to_le_bytes().to_vec(),
        _ => vec![0u8; elem_size],
    }
}

pub fn int_scalar_to_le_bytes(scalar: Scalar, elem_size: usize) -> Vec<u8> {
    match elem_size {
        1 => vec![scalar.elem::<i8>() as u8],
        2 => scalar.elem::<i16>().to_le_bytes().to_vec(),
        4 => scalar.elem::<i32>().to_le_bytes().to_vec(),
        8 => scalar.elem::<i64>().to_le_bytes().to_vec(),
        _ => vec![0u8; elem_size],
    }
}

#[derive(Clone, Copy)]
pub enum CmpOp {
    Equal,
    Greater,
    GreaterEqual,
    Lower,
    LowerEqual,
}

fn eval_cmp_f32(diff: &[f32], op: CmpOp) -> Vec<u8> {
    const EPS: f32 = 1e-3;
    diff.iter()
        .map(|&d| {
            let b = match op {
                CmpOp::Equal => d.abs() <= EPS,
                CmpOp::Greater => d > EPS,
                CmpOp::GreaterEqual => d >= -EPS,
                CmpOp::Lower => d < -EPS,
                CmpOp::LowerEqual => d <= EPS,
            };
            u8::from(b)
        })
        .collect()
}

fn eval_cmp_i64(diff: &[i64], op: CmpOp) -> Vec<u8> {
    diff.iter()
        .map(|&d| {
            let b = match op {
                CmpOp::Equal => d == 0,
                CmpOp::Greater => d > 0,
                CmpOp::GreaterEqual => d >= 0,
                CmpOp::Lower => d < 0,
                CmpOp::LowerEqual => d <= 0,
            };
            u8::from(b)
        })
        .collect()
}

pub fn compare_float(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
    op: CmpOp,
) -> GnaBoolTensorPrimitive {
    let diff = ops::execute_float_add_sub(lhs, rhs, -1);
    let shape = diff.shape();
    let vals = decode_floats(diff.buffer().as_slice(), diff.dtype());
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(eval_cmp_f32(&vals, op)))
}

pub fn compare_float_elem(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
    op: CmpOp,
) -> GnaBoolTensorPrimitive {
    let diff = ops::execute_float_add_scalar(lhs, rhs, -1.0);
    let shape = diff.shape();
    let vals = decode_floats(diff.buffer().as_slice(), diff.dtype());
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(eval_cmp_f32(&vals, op)))
}

pub fn compare_int(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
    op: CmpOp,
) -> GnaBoolTensorPrimitive {
    let diff = ops::execute_int_add_sub(lhs, rhs, -1);
    let shape = diff.shape();
    let vals = decode_ints(diff.buffer().as_slice(), diff.dtype());
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(eval_cmp_i64(&vals, op)))
}

pub fn compare_int_elem(
    lhs: GnaIntTensorPrimitive,
    rhs: Scalar,
    op: CmpOp,
) -> GnaBoolTensorPrimitive {
    let diff = ops::execute_int_add_scalar(lhs, rhs, -1);
    let shape = diff.shape();
    let vals = decode_ints(diff.buffer().as_slice(), diff.dtype());
    GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(eval_cmp_i64(&vals, op)))
}

pub fn convert_float_to_int(
    tensor: GnaFloatTensorPrimitive,
    out_dtype: burn_backend::IntDType,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let vals: Vec<i64> = decode_floats(tensor.buffer().as_slice(), tensor.dtype())
        .into_iter()
        .map(|v| v as i64)
        .collect();
    GnaIntTensorPrimitive::new(
        shape,
        out_dtype,
        GnaBuffer::from_bytes(encode_ints(&vals, out_dtype)),
    )
}

pub fn convert_int_to_float(
    tensor: GnaIntTensorPrimitive,
    out_dtype: burn_backend::FloatDType,
) -> GnaFloatTensorPrimitive {
    let shape = tensor.shape();
    let vals: Vec<f32> = decode_ints(tensor.buffer().as_slice(), tensor.dtype())
        .into_iter()
        .map(|v| v as f32)
        .collect();
    GnaFloatTensorPrimitive::new(
        shape,
        out_dtype,
        GnaBuffer::from_bytes(encode_floats(&vals, out_dtype)),
    )
}
