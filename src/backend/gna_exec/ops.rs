use burn_backend::ops::ConvOptions;
use burn_backend::{Scalar, Shape, TensorMetadata};

use super::{
    GnaAffineOperands, GnaMatmulDims, decode_floats, decode_ints, encode_floats, encode_ints,
    gna_affine_i16, gna_matmul_i16,
};
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use crate::{GnaBackendError, Result};

fn to_i16_checked(values: &[i64]) -> Vec<i16> {
    values
        .iter()
        .map(|&v| i16::try_from(v).map_err(|_| GnaBackendError::IntValueOutOfI16Range(v)))
        .collect::<Result<Vec<_>>>()
        .expect("GNA hardware integer value out of Int16 range")
}

fn scalar_to_i64(s: Scalar) -> i64 {
    s.elem::<i64>()
}

fn scalar_to_f32(s: Scalar) -> f32 {
    s.elem::<f32>()
}

struct I16Pair<'a> {
    lhs: &'a [i16],
    rhs: &'a [i16],
}

fn gna_add_sub_i16(pair: I16Pair<'_>, sign: i16) -> Vec<i32> {
    let I16Pair { lhs, rhs } = pair;
    let total = lhs.len();
    if total == 0 {
        return Vec::new();
    }
    let m = total.div_ceil(32);
    let mut packed = vec![0i16; m * 64];
    for i in 0..total {
        packed[i * 2] = lhs[i];
        packed[i * 2 + 1] = rhs[i % rhs.len()];
    }
    let mut weights = vec![0i16; 64 * 32];
    for j in 0..32 {
        weights[(2 * j) * 32 + j] = 1;
        weights[(2 * j + 1) * 32 + j] = sign;
    }
    let mut out = gna_matmul_i16(&packed, &weights, GnaMatmulDims::new(m, 64, 32, 0))
        .expect("GNA hardware add/sub execution failed");
    out.truncate(total);
    out
}

fn gna_add_scalar_i16(lhs: &[i16], scalar_bias: i32) -> Vec<i32> {
    let total = lhs.len();
    if total == 0 {
        return Vec::new();
    }
    let m = total.div_ceil(64);
    let mut packed = vec![0i16; m * 64];
    packed[..total].copy_from_slice(lhs);

    let mut identity = vec![0i16; 64 * 64];
    for i in 0..64 {
        identity[i * 64 + i] = 1;
    }
    let bias = vec![scalar_bias; 64];
    let mut out = gna_affine_i16(
        GnaAffineOperands {
            lhs: &packed,
            rhs: &identity,
            bias: Some(&bias),
        },
        GnaMatmulDims::new(m, 64, 64, 0),
    )
    .expect("GNA hardware add_scalar execution failed");
    out.truncate(total);
    out
}

fn gna_mul_scalar_i16(lhs: &[i16], scalar: i16) -> Vec<i32> {
    let total = lhs.len();
    if total == 0 {
        return Vec::new();
    }
    let m = total.div_ceil(64);
    let mut packed = vec![0i16; m * 64];
    packed[..total].copy_from_slice(lhs);

    let mut diag = vec![0i16; 64 * 64];
    for i in 0..64 {
        diag[i * 64 + i] = scalar;
    }
    let mut out = gna_matmul_i16(&packed, &diag, GnaMatmulDims::new(m, 64, 64, 0))
        .expect("GNA hardware mul_scalar execution failed");
    out.truncate(total);
    out
}

fn gna_mul_i16(lhs: &[i16], rhs: &[i16]) -> Vec<i32> {
    let total = lhs.len();
    let mut out = Vec::with_capacity(total);
    for offset in (0..total).step_by(64) {
        let len = (total - offset).min(64);
        let mut input = vec![0i16; 64];
        input[..len].copy_from_slice(&lhs[offset..offset + len]);
        let mut diag = vec![0i16; 64 * 64];
        for i in 0..len {
            diag[i * 64 + i] = rhs[(offset + i) % rhs.len()];
        }
        let res = gna_matmul_i16(&input, &diag, GnaMatmulDims::new(1, 64, 64, 0))
            .expect("GNA hardware mul execution failed");
        out.extend_from_slice(&res[..len]);
    }
    out
}

fn gna_sum_i16(values: &[i16]) -> i32 {
    if values.is_empty() {
        return 0;
    }
    let m = values.len().div_ceil(64);
    let mut packed = vec![0i16; m * 64];
    packed[..values.len()].copy_from_slice(values);
    let ones = vec![1i16; 64];
    let partials = gna_matmul_i16(&packed, &ones, GnaMatmulDims::new(m, 64, 1, 0))
        .expect("GNA hardware sum execution failed");
    partials.into_iter().sum()
}

pub fn execute_int_add_sub(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
    sign: i16,
) -> GnaIntTensorPrimitive {
    let shape = lhs.shape();
    let dtype = lhs.dtype();
    let l = to_i16_checked(&decode_ints(lhs.buffer().as_slice(), dtype));
    let r = to_i16_checked(&decode_ints(rhs.buffer().as_slice(), rhs.dtype()));
    let out: Vec<i64> = gna_add_sub_i16(I16Pair { lhs: &l, rhs: &r }, sign)
        .into_iter()
        .map(|v| v as i64)
        .collect();
    GnaIntTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_ints(&out, dtype)),
    )
}

pub fn execute_int_add_scalar(
    lhs: GnaIntTensorPrimitive,
    rhs: Scalar,
    sign: i32,
) -> GnaIntTensorPrimitive {
    let shape = lhs.shape();
    let dtype = lhs.dtype();
    let l = to_i16_checked(&decode_ints(lhs.buffer().as_slice(), dtype));
    let s = (scalar_to_i64(rhs) as i32) * sign;
    let out: Vec<i64> = gna_add_scalar_i16(&l, s)
        .into_iter()
        .map(|v| v as i64)
        .collect();
    GnaIntTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_ints(&out, dtype)),
    )
}

pub fn execute_int_mul(
    lhs: GnaIntTensorPrimitive,
    rhs: GnaIntTensorPrimitive,
) -> GnaIntTensorPrimitive {
    let shape = lhs.shape();
    let dtype = lhs.dtype();
    let l = to_i16_checked(&decode_ints(lhs.buffer().as_slice(), dtype));
    let r = to_i16_checked(&decode_ints(rhs.buffer().as_slice(), rhs.dtype()));
    let out: Vec<i64> = gna_mul_i16(&l, &r).into_iter().map(|v| v as i64).collect();
    GnaIntTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_ints(&out, dtype)),
    )
}

pub fn execute_int_mul_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
    let shape = lhs.shape();
    let dtype = lhs.dtype();
    let l = to_i16_checked(&decode_ints(lhs.buffer().as_slice(), dtype));
    let s = i16::try_from(scalar_to_i64(rhs)).expect("GNA scalar out of Int16 range");
    let out: Vec<i64> = gna_mul_scalar_i16(&l, s)
        .into_iter()
        .map(|v| v as i64)
        .collect();
    GnaIntTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_ints(&out, dtype)),
    )
}

fn quantize_pair(l: &[f32], r: &[f32]) -> (Vec<i16>, Vec<i16>, f32) {
    let max_abs = l
        .iter()
        .chain(r.iter())
        .fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let q_max = 16384.0f32;
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / q_max };
    let l_q = l
        .iter()
        .map(|&v| (v / scale).round().clamp(-q_max, q_max) as i16)
        .collect();
    let r_q = r
        .iter()
        .map(|&v| (v / scale).round().clamp(-q_max, q_max) as i16)
        .collect();
    (l_q, r_q, scale)
}

pub fn execute_float_add_sub(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
    sign: i16,
) -> GnaFloatTensorPrimitive {
    let dtype = lhs.dtype();
    let elem_size = burn_backend::DType::from(dtype).size();
    let (shape, l_bytes, r_bytes) = super::layout::broadcast_pair_bytes(
        super::layout::shape::ByteTensorView::new(lhs.buffer().as_slice(), &lhs.shape(), elem_size),
        super::layout::shape::ByteTensorView::new(rhs.buffer().as_slice(), &rhs.shape(), elem_size),
    );
    let l = decode_floats(&l_bytes, dtype);
    let r = decode_floats(&r_bytes, rhs.dtype());
    let (l_q, r_q, scale) = quantize_pair(&l, &r);
    let out: Vec<f32> = gna_add_sub_i16(
        I16Pair {
            lhs: &l_q,
            rhs: &r_q,
        },
        sign,
    )
    .into_iter()
    .map(|v| v as f32 * scale)
    .collect();
    GnaFloatTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub fn execute_float_add_scalar(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
    sign: f32,
) -> GnaFloatTensorPrimitive {
    let shape = lhs.shape();
    let dtype = lhs.dtype();
    let l = decode_floats(lhs.buffer().as_slice(), dtype);
    let s = scalar_to_f32(rhs) * sign;
    let max_abs = l.iter().fold(s.abs(), |acc, &v| acc.max(v.abs()));
    let q_max = 16384.0f32;
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / q_max };
    let l_q: Vec<i16> = l
        .iter()
        .map(|&v| (v / scale).round().clamp(-q_max, q_max) as i16)
        .collect();
    let s_q = (s / scale).round() as i32;
    let out: Vec<f32> = gna_add_scalar_i16(&l_q, s_q)
        .into_iter()
        .map(|v| v as f32 * scale)
        .collect();
    GnaFloatTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub fn execute_float_mul(
    lhs: GnaFloatTensorPrimitive,
    rhs: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    let dtype = lhs.dtype();
    let elem_size = burn_backend::DType::from(dtype).size();
    let (shape, l_bytes, r_bytes) = super::layout::broadcast_pair_bytes(
        super::layout::shape::ByteTensorView::new(lhs.buffer().as_slice(), &lhs.shape(), elem_size),
        super::layout::shape::ByteTensorView::new(rhs.buffer().as_slice(), &rhs.shape(), elem_size),
    );
    let l = decode_floats(&l_bytes, dtype);
    let r = decode_floats(&r_bytes, rhs.dtype());
    let max_l = l.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let max_r = r.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let q_max = 16384.0f32;
    let scale_l = if max_l == 0.0 { 1.0 } else { max_l / q_max };
    let scale_r = if max_r == 0.0 { 1.0 } else { max_r / q_max };
    let l_q: Vec<i16> = l
        .iter()
        .map(|&v| (v / scale_l).round().clamp(-q_max, q_max) as i16)
        .collect();
    let r_q: Vec<i16> = r
        .iter()
        .map(|&v| (v / scale_r).round().clamp(-q_max, q_max) as i16)
        .collect();
    let combined = scale_l * scale_r;
    let out: Vec<f32> = gna_mul_i16(&l_q, &r_q)
        .into_iter()
        .map(|v| v as f32 * combined)
        .collect();
    GnaFloatTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub fn execute_float_mul_scalar(
    lhs: GnaFloatTensorPrimitive,
    rhs: Scalar,
) -> GnaFloatTensorPrimitive {
    let shape = lhs.shape();
    let dtype = lhs.dtype();
    let l = decode_floats(lhs.buffer().as_slice(), dtype);
    let s = scalar_to_f32(rhs);
    let max_l = l.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let q_max = 16384.0f32;
    let scale_l = if max_l == 0.0 { 1.0 } else { max_l / q_max };
    let scale_s = if s == 0.0 { 1.0 } else { s.abs() / q_max };
    let l_q: Vec<i16> = l
        .iter()
        .map(|&v| (v / scale_l).round().clamp(-q_max, q_max) as i16)
        .collect();
    let s_q = (s / scale_s).round().clamp(-q_max, q_max) as i16;
    let combined = scale_l * scale_s;
    let out: Vec<f32> = gna_mul_scalar_i16(&l_q, s_q)
        .into_iter()
        .map(|v| v as f32 * combined)
        .collect();
    GnaFloatTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub fn execute_int_sum(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
    let dtype = tensor.dtype();
    let vals = to_i16_checked(&decode_ints(tensor.buffer().as_slice(), dtype));
    let total = gna_sum_i16(&vals) as i64;
    GnaIntTensorPrimitive::new(
        Shape::from([1]),
        dtype,
        GnaBuffer::from_bytes(encode_ints(&[total], dtype)),
    )
}

pub fn execute_float_sum(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    let dtype = tensor.dtype();
    let vals = decode_floats(tensor.buffer().as_slice(), dtype);
    let max_abs = vals.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let q_max = 16384.0f32;
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / q_max };
    let q: Vec<i16> = vals
        .iter()
        .map(|&v| (v / scale).round().clamp(-q_max, q_max) as i16)
        .collect();
    let total = gna_sum_i16(&q) as f32 * scale;
    GnaFloatTensorPrimitive::new(
        Shape::from([1]),
        dtype,
        GnaBuffer::from_bytes(encode_floats(&[total], dtype)),
    )
}

fn reduce_dim_indices(dims: &[usize], dim: usize) -> (Shape, usize, usize, usize) {
    let mut out_dims = dims.to_vec();
    out_dims[dim] = 1;
    let outer: usize = dims[..dim].iter().product();
    let dim_len = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    (Shape::from(out_dims), outer, dim_len, inner)
}

pub fn execute_int_sum_dim(
    tensor: GnaIntTensorPrimitive,
    dim: usize,
    mean: bool,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let dtype = tensor.dtype();
    let vals = to_i16_checked(&decode_ints(tensor.buffer().as_slice(), dtype));
    let (out_shape, outer, dim_len, inner) = reduce_dim_indices(dims, dim);

    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let slice: Vec<i16> = (0..dim_len)
                .map(|d| vals[(o * dim_len + d) * inner + i])
                .collect();
            let s = gna_sum_i16(&slice) as i64;
            out.push(if mean { s / dim_len as i64 } else { s });
        }
    }
    GnaIntTensorPrimitive::new(
        out_shape,
        dtype,
        GnaBuffer::from_bytes(encode_ints(&out, dtype)),
    )
}

pub fn execute_float_sum_dim(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
    mean: bool,
) -> GnaFloatTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let dtype = tensor.dtype();
    let vals = decode_floats(tensor.buffer().as_slice(), dtype);
    let (out_shape, outer, dim_len, inner) = reduce_dim_indices(dims, dim);

    let max_abs = vals.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let q_max = 16384.0f32;
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / q_max };

    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let slice: Vec<i16> = (0..dim_len)
                .map(|d| {
                    (vals[(o * dim_len + d) * inner + i] / scale)
                        .round()
                        .clamp(-q_max, q_max) as i16
                })
                .collect();
            let s = gna_sum_i16(&slice) as f32 * scale;
            out.push(if mean { s / dim_len as f32 } else { s });
        }
    }
    GnaFloatTensorPrimitive::new(
        out_shape,
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

/// Execute 2D convolution (`im2col` + GNA hardware `FullyConnectedAffine`).
pub fn execute_float_conv2d(
    x: GnaFloatTensorPrimitive,
    weight: GnaFloatTensorPrimitive,
    bias: Option<GnaFloatTensorPrimitive>,
    options: ConvOptions<2>,
) -> GnaFloatTensorPrimitive {
    let x_shape = x.shape();
    let w_shape = weight.shape();
    let xd: &[usize] = &x_shape;
    let wd: &[usize] = &w_shape;
    let (batch, in_c, in_h, in_w) = (xd[0], xd[1], xd[2], xd[3]);
    let (out_c, in_c_per_g, kh, kw) = (wd[0], wd[1], wd[2], wd[3]);
    let [sh, sw] = options.stride;
    let [(ph_top, ph_bot), (pw_left, pw_right)] = options.padding;
    let [dh, dw] = options.dilation;
    let groups = options.groups;

    let out_h = (in_h + ph_top + ph_bot - dh * (kh - 1) - 1) / sh + 1;
    let out_w = (in_w + pw_left + pw_right - dw * (kw - 1) - 1) / sw + 1;

    let dtype = x.dtype();
    let x_vals = decode_floats(x.buffer().as_slice(), dtype);
    let w_vals = decode_floats(weight.buffer().as_slice(), weight.dtype());
    let b_vals = bias.map(|b| decode_floats(b.buffer().as_slice(), b.dtype()));

    let k_col = in_c_per_g * kh * kw;
    let out_c_per_g = out_c / groups;
    let q_max = ((i32::MAX as f64 / (k_col.max(1) as f64))
        .sqrt()
        .min(16384.0)
        .floor()) as f32;

    let max_x = x_vals.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let max_w = w_vals.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let scale_x = if max_x == 0.0 { 1.0 } else { max_x / q_max };
    let scale_w = if max_w == 0.0 { 1.0 } else { max_w / q_max };
    let combined_scale = scale_x * scale_w;

    let mut out_vals = vec![0.0f32; batch * out_c * out_h * out_w];

    for g in 0..groups {
        let m = batch * out_h * out_w;
        let mut col_i16 = vec![0i16; m * k_col];
        for b in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let row_idx = (b * out_h + oh) * out_w + ow;
                    for ic in 0..in_c_per_g {
                        let c_abs = g * in_c_per_g + ic;
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (oh * sh + ky * dh) as isize - ph_top as isize;
                                let ix = (ow * sw + kx * dw) as isize - pw_left as isize;
                                let val = if iy >= 0
                                    && iy < in_h as isize
                                    && ix >= 0
                                    && ix < in_w as isize
                                {
                                    x_vals[((b * in_c + c_abs) * in_h + iy as usize) * in_w
                                        + ix as usize]
                                } else {
                                    0.0
                                };
                                let col_idx = (ic * kh + ky) * kw + kx;
                                col_i16[row_idx * k_col + col_idx] =
                                    (val / scale_x).round().clamp(-q_max, q_max) as i16;
                            }
                        }
                    }
                }
            }
        }

        let mut w_i16 = vec![0i16; k_col * out_c_per_g];
        for oc in 0..out_c_per_g {
            let oc_abs = g * out_c_per_g + oc;
            for k_idx in 0..k_col {
                let val = w_vals[oc_abs * k_col + k_idx];
                w_i16[k_idx * out_c_per_g + oc] =
                    (val / scale_w).round().clamp(-q_max, q_max) as i16;
            }
        }

        let b_i32 = b_vals.as_ref().map(|bv| {
            (0..out_c_per_g)
                .map(|oc| (bv[g * out_c_per_g + oc] / combined_scale).round() as i32)
                .collect::<Vec<_>>()
        });

        let gna_out = gna_affine_i16(
            GnaAffineOperands {
                lhs: &col_i16,
                rhs: &w_i16,
                bias: b_i32.as_deref(),
            },
            GnaMatmulDims::new(m, k_col, out_c_per_g, 0),
        )
        .expect("GNA hardware conv2d execution failed");

        for b in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let row_idx = (b * out_h + oh) * out_w + ow;
                    for oc in 0..out_c_per_g {
                        let oc_abs = g * out_c_per_g + oc;
                        out_vals[((b * out_c + oc_abs) * out_h + oh) * out_w + ow] =
                            gna_out[row_idx * out_c_per_g + oc] as f32 * combined_scale;
                    }
                }
            }
        }
    }

    GnaFloatTensorPrimitive::new(
        Shape::from([batch, out_c, out_h, out_w]),
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out_vals, dtype)),
    )
}
