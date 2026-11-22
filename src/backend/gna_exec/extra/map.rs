use burn_backend::{Scalar, TensorMetadata};

use crate::backend::gna_exec::{decode_floats, decode_ints, encode_floats, encode_ints, layout};
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;

pub fn map_float_unary(
    tensor: GnaFloatTensorPrimitive,
    f: impl Fn(f32) -> f32,
) -> GnaFloatTensorPrimitive {
    let dtype = tensor.dtype();
    let out: Vec<f32> = decode_floats(tensor.buffer().as_slice(), dtype)
        .into_iter()
        .map(f)
        .collect();
    GnaFloatTensorPrimitive::new(
        tensor.shape(),
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub fn map_float_binary(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
    f: impl Fn(f32, f32) -> f32,
) -> GnaFloatTensorPrimitive {
    let dtype = lhs.dtype();
    let elem_size = burn_backend::DType::from(dtype).size();
    let (shape, l_bytes, r_bytes) = layout::broadcast_pair_bytes(
        layout::shape::ByteTensorView::new(lhs.buffer().as_slice(), &lhs.shape(), elem_size),
        layout::shape::ByteTensorView::new(rhs.buffer().as_slice(), &rhs.shape(), elem_size),
    );
    let l = decode_floats(&l_bytes, dtype);
    let r = decode_floats(&r_bytes, rhs.dtype());
    let out: Vec<f32> = (0..l.len()).map(|i| f(l[i], r[i])).collect();
    GnaFloatTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub fn map_float_scalar(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
    f: impl Fn(f32, f32) -> f32,
) -> GnaFloatTensorPrimitive {
    let s = rhs.elem::<f32>();
    map_float_unary(lhs, |v| f(v, s))
}

pub fn map_int_unary(
    tensor: GnaIntTensorPrimitive,
    f: impl Fn(i64) -> i64,
) -> GnaIntTensorPrimitive {
    let dtype = tensor.dtype();
    let out: Vec<i64> = decode_ints(tensor.buffer().as_slice(), dtype)
        .into_iter()
        .map(f)
        .collect();
    GnaIntTensorPrimitive::new(
        tensor.shape(),
        dtype,
        GnaBuffer::from_bytes(encode_ints(&out, dtype)),
    )
}

pub fn map_int_binary(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
    f: impl Fn(i64, i64) -> i64,
) -> GnaIntTensorPrimitive {
    let dtype = lhs.dtype();
    let elem_size = burn_backend::DType::from(dtype).size();
    let (shape, l_bytes, r_bytes) = layout::broadcast_pair_bytes(
        layout::shape::ByteTensorView::new(lhs.buffer().as_slice(), &lhs.shape(), elem_size),
        layout::shape::ByteTensorView::new(rhs.buffer().as_slice(), &rhs.shape(), elem_size),
    );
    let l = decode_ints(&l_bytes, dtype);
    let r = decode_ints(&r_bytes, rhs.dtype());
    let out: Vec<i64> = (0..l.len()).map(|i| f(l[i], r[i])).collect();
    GnaIntTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_ints(&out, dtype)),
    )
}

pub fn map_int_scalar(
    lhs: GnaIntTensorPrimitive,
    rhs: Scalar,
    f: impl Fn(i64, i64) -> i64,
) -> GnaIntTensorPrimitive {
    let s = rhs.elem::<i64>();
    map_int_unary(lhs, |v| f(v, s))
}

pub fn execute_float_cross(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
    dim: usize,
) -> GnaFloatTensorPrimitive {
    let shape = lhs.shape();
    let dims: &[usize] = &shape;
    let dtype = lhs.dtype();
    let l = decode_floats(lhs.buffer().as_slice(), dtype);
    let r = decode_floats(rhs.buffer().as_slice(), rhs.dtype());
    let outer: usize = dims[..dim].iter().product();
    let inner: usize = dims[dim + 1..].iter().product();
    let mut out = vec![0.0f32; l.len()];

    for o in 0..outer {
        for i in 0..inner {
            let idx0 = (o * 3) * inner + i;
            let idx1 = (o * 3 + 1) * inner + i;
            let idx2 = (o * 3 + 2) * inner + i;
            let (a0, a1, a2) = (l[idx0], l[idx1], l[idx2]);
            let (b0, b1, b2) = (r[idx0], r[idx1], r[idx2]);
            out[idx0] = a1 * b2 - a2 * b1;
            out[idx1] = a2 * b0 - a0 * b2;
            out[idx2] = a0 * b1 - a1 * b0;
        }
    }
    GnaFloatTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}
