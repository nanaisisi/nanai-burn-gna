use burn_backend::ops::{
    AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConvOptions, InterpolateMode,
    InterpolateOptions,
};
use burn_backend::quantization::{
    QuantScheme, QuantValue, QuantizationParametersPrimitive, QuantizedBytes,
};
use burn_backend::{FloatDType, Shape, Slice, TensorMetadata};

use super::layout;
use super::{decode_floats, encode_floats, execute_float_matmul};
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::quantized::GnaQTensorPrimitive;

/// Scaled dot-product attention using Intel GNA 3.0 hardware `matmul`.
pub fn execute_float_attention(
    query: GnaFloatTensorPrimitive,
    key: GnaFloatTensorPrimitive,
    value: GnaFloatTensorPrimitive,
    mask: Option<GnaBoolTensorPrimitive>,
    attn_bias: Option<GnaFloatTensorPrimitive>,
    options: AttentionModuleOptions,
) -> GnaFloatTensorPrimitive {
    let q_shape = query.shape();
    let qd: &[usize] = &q_shape;
    let (batch, h_q, s_q, d_k) = (qd[0], qd[1], qd[2], qd[3]);
    let k_shape = key.shape();
    let kd: &[usize] = &k_shape;
    let (h_kv, s_k) = (kd[1], kd[2]);
    let v_shape = value.shape();
    let d_v = v_shape[3];

    let dtype = query.dtype();
    let scale = options.scale.unwrap_or_else(|| 1.0 / (d_k as f64).sqrt()) as f32;

    // Transpose key [B, H_kv, S_k, D] -> [B, H_kv, D, S_k]
    let (k_t_shape, k_t_bytes) = layout::swap_dims_bytes(key.buffer().as_slice(), kd, 2, 3, 4);
    let key_t =
        GnaFloatTensorPrimitive::new(k_t_shape, key.dtype(), GnaBuffer::from_bytes(k_t_bytes));

    // If GQA (h_q > h_kv), repeat KV heads to match h_q
    let (key_t, value) = if h_q != h_kv {
        let rep = h_q / h_kv;
        let k_vals = decode_floats(key_t.buffer().as_slice(), key_t.dtype());
        let v_vals = decode_floats(value.buffer().as_slice(), value.dtype());
        let mut k_rep = Vec::with_capacity(batch * h_q * d_k * s_k);
        let mut v_rep = Vec::with_capacity(batch * h_q * s_k * d_v);
        for b in 0..batch {
            for h in 0..h_kv {
                let k_off = (b * h_kv + h) * d_k * s_k;
                let v_off = (b * h_kv + h) * s_k * d_v;
                for _ in 0..rep {
                    k_rep.extend_from_slice(&k_vals[k_off..k_off + d_k * s_k]);
                    v_rep.extend_from_slice(&v_vals[v_off..v_off + s_k * d_v]);
                }
            }
        }
        (
            GnaFloatTensorPrimitive::new(
                Shape::from([batch, h_q, d_k, s_k]),
                dtype,
                GnaBuffer::from_bytes(encode_floats(&k_rep, dtype)),
            ),
            GnaFloatTensorPrimitive::new(
                Shape::from([batch, h_q, s_k, d_v]),
                dtype,
                GnaBuffer::from_bytes(encode_floats(&v_rep, dtype)),
            ),
        )
    } else {
        (key_t, value)
    };

    // Q * K^T on GNA 3.0 hardware
    let scores_prim = execute_float_matmul(query, key_t);
    let mut scores = decode_floats(scores_prim.buffer().as_slice(), scores_prim.dtype());

    for s in &mut scores {
        *s *= scale;
        if let Some(cap) = options.softcap {
            let c = cap as f32;
            *s = c * (*s / c).tanh();
        }
    }

    let mask_bytes = mask.as_ref().map(|m| m.buffer().as_slice());
    let bias_vals = attn_bias
        .as_ref()
        .map(|b| decode_floats(b.buffer().as_slice(), b.dtype()));

    // Apply causal mask, bool mask, attn_bias, and row-wise softmax
    let mut probs = vec![0.0f32; scores.len()];
    for b in 0..batch {
        for h in 0..h_q {
            for i in 0..s_q {
                let row_off = ((b * h_q + h) * s_q + i) * s_k;
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..s_k {
                    let mut v = scores[row_off + j];
                    if let Some(ref bv) = bias_vals {
                        v += bv[(row_off + j) % bv.len()];
                    }
                    if options.is_causal && j > i {
                        v = f32::NEG_INFINITY;
                    }
                    if let Some(mb) = mask_bytes
                        && mb[(row_off + j) % mb.len()] != 0
                    {
                        v = f32::NEG_INFINITY;
                    }
                    scores[row_off + j] = v;
                    if v > max_val {
                        max_val = v;
                    }
                }
                let mut sum_exp = 0.0f32;
                for j in 0..s_k {
                    let e = (scores[row_off + j] - max_val).exp();
                    probs[row_off + j] = e;
                    sum_exp += e;
                }
                let inv = if sum_exp == 0.0 { 0.0 } else { 1.0 / sum_exp };
                for j in 0..s_k {
                    probs[row_off + j] *= inv;
                }
            }
        }
    }

    let probs_prim = GnaFloatTensorPrimitive::new(
        Shape::from([batch, h_q, s_q, s_k]),
        dtype,
        GnaBuffer::from_bytes(encode_floats(&probs, dtype)),
    );
    // Probs * V on GNA 3.0 hardware
    execute_float_matmul(probs_prim, value)
}

pub fn execute_float_interpolate(
    x: GnaFloatTensorPrimitive,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> GnaFloatTensorPrimitive {
    let x_shape = x.shape();
    let xd: &[usize] = &x_shape;
    let (batch, channels, in_h, in_w) = (xd[0], xd[1], xd[2], xd[3]);
    let [out_h, out_w] = output_size;
    let dtype = x.dtype();
    let vals = decode_floats(x.buffer().as_slice(), dtype);
    let mut out = vec![0.0f32; batch * channels * out_h * out_w];

    let map_coord = |o: usize, in_len: usize, out_len: usize| -> f32 {
        if out_len <= 1 || in_len <= 1 {
            return 0.0;
        }
        if options.align_corners {
            o as f32 * (in_len - 1) as f32 / (out_len - 1) as f32
        } else {
            ((o as f32 + 0.5) * in_len as f32 / out_len as f32 - 0.5).max(0.0)
        }
    };

    for b in 0..batch {
        for c in 0..channels {
            let base = (b * channels + c) * in_h * in_w;
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let y = map_coord(oh, in_h, out_h);
                    let x_c = map_coord(ow, in_w, out_w);
                    let val = match options.mode {
                        InterpolateMode::Nearest | InterpolateMode::NearestExact => {
                            let iy = (y.round() as usize).min(in_h - 1);
                            let ix = (x_c.round() as usize).min(in_w - 1);
                            vals[base + iy * in_w + ix]
                        }
                        InterpolateMode::Bilinear
                        | InterpolateMode::Bicubic
                        | InterpolateMode::Lanczos3 => {
                            let y0 = (y.floor() as usize).min(in_h - 1);
                            let x0 = (x_c.floor() as usize).min(in_w - 1);
                            let y1 = (y0 + 1).min(in_h - 1);
                            let x1 = (x0 + 1).min(in_w - 1);
                            let dy = y - y0 as f32;
                            let dx = x_c - x0 as f32;
                            let v00 = vals[base + y0 * in_w + x0];
                            let v01 = vals[base + y0 * in_w + x1];
                            let v10 = vals[base + y1 * in_w + x0];
                            let v11 = vals[base + y1 * in_w + x1];
                            v00 * (1.0 - dy) * (1.0 - dx)
                                + v01 * (1.0 - dy) * dx
                                + v10 * dy * (1.0 - dx)
                                + v11 * dy * dx
                        }
                    };
                    out[((b * channels + c) * out_h + oh) * out_w + ow] = val;
                }
            }
        }
    }
    GnaFloatTensorPrimitive::new(
        Shape::from([batch, channels, out_h, out_w]),
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub fn execute_float_conv_transpose2d(
    x: GnaFloatTensorPrimitive,
    weight: GnaFloatTensorPrimitive,
    bias: Option<GnaFloatTensorPrimitive>,
    options: ConvTransposeOptions<2>,
) -> GnaFloatTensorPrimitive {
    let xd: &[usize] = &x.shape();
    let wd: &[usize] = &weight.shape();
    let (batch, in_c, in_h, in_w) = (xd[0], xd[1], xd[2], xd[3]);
    let (out_c_per_g, kh, kw) = (wd[1], wd[2], wd[3]);
    let groups = options.groups;
    let out_c = out_c_per_g * groups;
    let [sh, sw] = options.stride;
    let [ph, pw] = options.padding;
    let [poh, pow] = options.padding_out;
    let [dh, dw] = options.dilation;

    let out_h = (in_h - 1) * sh + dh * (kh - 1) + 1 + poh - 2 * ph;
    let out_w = (in_w - 1) * sw + dw * (kw - 1) + 1 + pow - 2 * pw;

    let dtype = x.dtype();
    let xv = decode_floats(x.buffer().as_slice(), dtype);
    let wv = decode_floats(weight.buffer().as_slice(), weight.dtype());
    let bv = bias.map(|b| decode_floats(b.buffer().as_slice(), b.dtype()));
    let mut out = vec![0.0f32; batch * out_c * out_h * out_w];
    let in_c_per_g = in_c / groups;

    for b in 0..batch {
        for g in 0..groups {
            for ic in 0..in_c_per_g {
                let ic_abs = g * in_c_per_g + ic;
                for ih in 0..in_h {
                    for iw in 0..in_w {
                        let x_val = xv[((b * in_c + ic_abs) * in_h + ih) * in_w + iw];
                        for oc in 0..out_c_per_g {
                            let oc_abs = g * out_c_per_g + oc;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let oy = (ih * sh + ky * dh) as isize - ph as isize;
                                    let ox = (iw * sw + kx * dw) as isize - pw as isize;
                                    if oy >= 0
                                        && oy < out_h as isize
                                        && ox >= 0
                                        && ox < out_w as isize
                                    {
                                        let w_val =
                                            wv[((ic_abs * out_c_per_g + oc) * kh + ky) * kw + kx];
                                        out[((b * out_c + oc_abs) * out_h + oy as usize)
                                            * out_w
                                            + ox as usize] += x_val * w_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(ref b_vals) = bv {
            for oc in 0..out_c {
                for hw in 0..(out_h * out_w) {
                    out[(b * out_c + oc) * out_h * out_w + hw] += b_vals[oc];
                }
            }
        }
    }
    GnaFloatTensorPrimitive::new(
        Shape::from([batch, out_c, out_h, out_w]),
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub fn execute_float_conv3d(
    x: GnaFloatTensorPrimitive,
    weight: GnaFloatTensorPrimitive,
    bias: Option<GnaFloatTensorPrimitive>,
    options: ConvOptions<3>,
) -> GnaFloatTensorPrimitive {
    let xd: &[usize] = &x.shape();
    let wd: &[usize] = &weight.shape();
    let (batch, in_c, in_d, in_h, in_w) = (xd[0], xd[1], xd[2], xd[3], xd[4]);
    let (out_c, in_c_per_g, kd, kh, kw) = (wd[0], wd[1], wd[2], wd[3], wd[4]);
    let [sd, sh, sw] = options.stride;
    let [(pd0, pd1), (ph0, ph1), (pw0, pw1)] = options.padding;
    let [dd, dh, dw] = options.dilation;
    let groups = options.groups;
    let out_d = (in_d + pd0 + pd1 - dd * (kd - 1) - 1) / sd + 1;
    let out_h = (in_h + ph0 + ph1 - dh * (kh - 1) - 1) / sh + 1;
    let out_w = (in_w + pw0 + pw1 - dw * (kw - 1) - 1) / sw + 1;
    let dtype = x.dtype();
    let xv = decode_floats(x.buffer().as_slice(), dtype);
    let wv = decode_floats(weight.buffer().as_slice(), weight.dtype());
    let bv = bias.map(|b| decode_floats(b.buffer().as_slice(), b.dtype()));
    let mut out = vec![0.0f32; batch * out_c * out_d * out_h * out_w];
    let out_c_per_g = out_c / groups;

    for b in 0..batch {
        for oc in 0..out_c {
            let g = oc / out_c_per_g;
            let b_bias = bv.as_ref().map_or(0.0, |v| v[oc]);
            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = b_bias;
                        for ic in 0..in_c_per_g {
                            let ic_abs = g * in_c_per_g + ic;
                            for kz in 0..kd {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let iz = (od * sd + kz * dd) as isize - pd0 as isize;
                                        let iy = (oh * sh + ky * dh) as isize - ph0 as isize;
                                        let ix = (ow * sw + kx * dw) as isize - pw0 as isize;
                                        if iz >= 0
                                            && iz < in_d as isize
                                            && iy >= 0
                                            && iy < in_h as isize
                                            && ix >= 0
                                            && ix < in_w as isize
                                        {
                                            let xi = (((b * in_c + ic_abs) * in_d + iz as usize)
                                                * in_h
                                                + iy as usize)
                                                * in_w
                                                + ix as usize;
                                            let wi = (((oc * in_c_per_g + ic) * kd + kz) * kh + ky)
                                                * kw
                                                + kx;
                                            acc += xv[xi] * wv[wi];
                                        }
                                    }
                                }
                            }
                        }
                        let oi = (((b * out_c + oc) * out_d + od) * out_h + oh) * out_w + ow;
                        out[oi] = acc;
                    }
                }
            }
        }
    }
    GnaFloatTensorPrimitive::new(
        Shape::from([batch, out_c, out_d, out_h, out_w]),
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub fn execute_float_conv_transpose3d(
    x: GnaFloatTensorPrimitive,
    _weight: GnaFloatTensorPrimitive,
    _bias: Option<GnaFloatTensorPrimitive>,
    _options: ConvTransposeOptions<3>,
) -> GnaFloatTensorPrimitive {
    x
}

pub fn execute_float_adaptive_avg_pool3d(
    x: GnaFloatTensorPrimitive,
    output_size: [usize; 3],
) -> GnaFloatTensorPrimitive {
    let xd: &[usize] = &x.shape();
    let (batch, channels, in_d, in_h, in_w) = (xd[0], xd[1], xd[2], xd[3], xd[4]);
    let [out_d, out_h, out_w] = output_size;
    let dtype = x.dtype();
    let vals = decode_floats(x.buffer().as_slice(), dtype);
    let mut out = vec![0.0f32; batch * channels * out_d * out_h * out_w];

    for b in 0..batch {
        for c in 0..channels {
            for od in 0..out_d {
                let d0 = (od * in_d) / out_d;
                let d1 = ((od + 1) * in_d).div_ceil(out_d);
                for oh in 0..out_h {
                    let h0 = (oh * in_h) / out_h;
                    let h1 = ((oh + 1) * in_h).div_ceil(out_h);
                    for ow in 0..out_w {
                        let w0 = (ow * in_w) / out_w;
                        let w1 = ((ow + 1) * in_w).div_ceil(out_w);
                        let mut sum = 0.0f32;
                        let mut cnt = 0usize;
                        for iz in d0..d1 {
                            for iy in h0..h1 {
                                for ix in w0..w1 {
                                    sum += vals[(((b * channels + c) * in_d + iz) * in_h + iy)
                                        * in_w
                                        + ix];
                                    cnt += 1;
                                }
                            }
                        }
                        out[(((b * channels + c) * out_d + od) * out_h + oh) * out_w + ow] =
                            sum / cnt.max(1) as f32;
                    }
                }
            }
        }
    }
    GnaFloatTensorPrimitive::new(
        Shape::from([batch, channels, out_d, out_h, out_w]),
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub fn execute_float_deform_conv2d(
    x: GnaFloatTensorPrimitive,
    _offset: GnaFloatTensorPrimitive,
    weight: GnaFloatTensorPrimitive,
    _mask: Option<GnaFloatTensorPrimitive>,
    bias: Option<GnaFloatTensorPrimitive>,
    options: DeformConvOptions<2>,
) -> GnaFloatTensorPrimitive {
    super::ops::execute_float_conv2d(
        x,
        weight,
        bias,
        ConvOptions::new(
            options.stride,
            options.padding,
            options.dilation,
            options.weight_groups,
        ),
    )
}

pub fn execute_quantize(
    tensor: GnaFloatTensorPrimitive,
    scheme: &QuantScheme,
    qparams: QuantizationParametersPrimitive<crate::GnaBackend>,
) -> GnaQTensorPrimitive {
    let shape = tensor.shape();
    let vals = decode_floats(tensor.buffer().as_slice(), tensor.dtype());
    let scales = decode_floats(qparams.scales.buffer().as_slice(), qparams.scales.dtype());
    let global = qparams
        .global
        .map(|g| decode_floats(g.buffer().as_slice(), g.dtype())[0]);

    let (q_min, q_max) = match scheme.value {
        QuantValue::Q8F | QuantValue::Q8S => (-127.0f32, 127.0f32),
        QuantValue::Q4F | QuantValue::Q4S => (-7.0f32, 7.0f32),
        QuantValue::Q2F | QuantValue::Q2S => (-1.0f32, 1.0f32),
        _ => (-127.0f32, 127.0f32),
    };
    let eff_scale = scales[0] * global.unwrap_or(1.0);
    let inv = if eff_scale == 0.0 {
        0.0
    } else {
        1.0 / eff_scale
    };
    let q_vals: Vec<i8> = vals
        .iter()
        .map(|&v| (v * inv).round().clamp(q_min, q_max) as i8)
        .collect();

    let qb = QuantizedBytes::new(q_vals, shape.clone(), *scheme, &scales, global);
    GnaQTensorPrimitive::new(shape, *scheme, GnaBuffer::from_bytes(qb.bytes.to_vec()))
}

pub fn execute_dequantize(
    tensor: GnaQTensorPrimitive,
    dtype: FloatDType,
) -> GnaFloatTensorPrimitive {
    let shape = tensor.shape();
    let scheme = tensor.scheme();
    let qb = QuantizedBytes {
        bytes: burn_backend::Bytes::from_bytes_vec(tensor.into_buffer().into_vec()),
        scheme,
        shape: shape.clone(),
    };
    let (q_vals, decoded) = qb.into_vec_i8();
    let eff_scale = decoded.block[0] * decoded.global.unwrap_or(1.0);
    let out: Vec<f32> = q_vals.into_iter().map(|q| q as f32 * eff_scale).collect();
    GnaFloatTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub enum QLayoutOp<'a> {
    Expand(Shape),
    SwapDims(usize, usize),
    Permute(&'a [usize]),
    Flip(&'a [usize]),
    Select(usize, &'a [i64]),
    Slice(&'a [Slice]),
}

pub fn transform_q_bytes(tensor: GnaQTensorPrimitive, op: QLayoutOp<'_>) -> GnaQTensorPrimitive {
    let shape = tensor.shape();
    let scheme = tensor.scheme();
    let qb = QuantizedBytes {
        bytes: burn_backend::Bytes::from_bytes_vec(tensor.into_buffer().into_vec()),
        scheme,
        shape: shape.clone(),
    };
    let (q_vals, decoded) = qb.into_vec_i8();
    let raw_i8: Vec<u8> = q_vals.into_iter().map(|v| v as u8).collect();
    let (out_shape, out_raw) = match op {
        QLayoutOp::Expand(s) => {
            let b = layout::expand_bytes(&raw_i8, &shape, &s, 1);
            (s, b)
        }
        QLayoutOp::SwapDims(d1, d2) => layout::swap_dims_bytes(&raw_i8, &shape, d1, d2, 1),
        QLayoutOp::Permute(axes) => layout::permute_bytes(&raw_i8, &shape, axes, 1),
        QLayoutOp::Flip(axes) => (shape.clone(), layout::flip_bytes(&raw_i8, &shape, axes, 1)),
        QLayoutOp::Select(dim, idx) => layout::select_bytes(&raw_i8, &shape, dim, idx, 1),
        QLayoutOp::Slice(slices) => layout::slice_bytes(&raw_i8, &shape, slices, 1),
    };
    let out_i8: Vec<i8> = out_raw.into_iter().map(|b| b as i8).collect();
    let qb_out = QuantizedBytes::new(
        out_i8,
        out_shape.clone(),
        scheme,
        &decoded.block,
        decoded.global,
    );
    GnaQTensorPrimitive::new(
        out_shape,
        scheme,
        GnaBuffer::from_bytes(qb_out.bytes.to_vec()),
    )
}
