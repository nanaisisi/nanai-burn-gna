use std::sync::atomic::{AtomicU64, Ordering};

use burn_backend::{Distribution, FloatDType, IntDType, Scalar, Shape, TensorMetadata};

use super::{decode_floats, decode_ints, encode_floats, encode_ints, gna_matmul_i16};
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;

#[derive(Clone, Copy)]
pub enum CumMode {
    Sum,
    Prod,
    Min,
    Max,
}

fn reduce_dim_shape(dims: &[usize], dim: usize) -> (Shape, usize, usize, usize) {
    let mut out_dims = dims.to_vec();
    out_dims[dim] = 1;
    let outer: usize = dims[..dim].iter().product();
    let dim_len = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    (Shape::from(out_dims), outer, dim_len, inner)
}

pub fn execute_int_argmax_argmin(
    tensor: GnaIntTensorPrimitive,
    dim: usize,
    is_max: bool,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let vals = decode_ints(tensor.buffer().as_slice(), tensor.dtype());
    let (out_shape, outer, dim_len, inner) = reduce_dim_shape(dims, dim);
    let mut out = Vec::with_capacity(outer * inner);

    for o in 0..outer {
        for i in 0..inner {
            let mut best_idx = 0usize;
            let mut best_val = vals[o * dim_len * inner + i];
            for d in 1..dim_len {
                let v = vals[(o * dim_len + d) * inner + i];
                if (is_max && v > best_val) || (!is_max && v < best_val) {
                    best_val = v;
                    best_idx = d;
                }
            }
            out.push(best_idx as i64);
        }
    }
    GnaIntTensorPrimitive::new(
        out_shape,
        IntDType::I32,
        GnaBuffer::from_bytes(encode_ints(&out, IntDType::I32)),
    )
}

pub fn execute_float_argmax_argmin(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
    out_dtype: IntDType,
    is_max: bool,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let vals = decode_floats(tensor.buffer().as_slice(), tensor.dtype());
    let (out_shape, outer, dim_len, inner) = reduce_dim_shape(dims, dim);
    let mut out = Vec::with_capacity(outer * inner);

    for o in 0..outer {
        for i in 0..inner {
            let mut best_idx = 0usize;
            let mut best_val = vals[o * dim_len * inner + i];
            for d in 1..dim_len {
                let v = vals[(o * dim_len + d) * inner + i];
                if (is_max && v > best_val) || (!is_max && v < best_val) {
                    best_val = v;
                    best_idx = d;
                }
            }
            out.push(best_idx as i64);
        }
    }
    GnaIntTensorPrimitive::new(
        out_shape,
        out_dtype,
        GnaBuffer::from_bytes(encode_ints(&out, out_dtype)),
    )
}

pub fn execute_int_argtopk(
    tensor: GnaIntTensorPrimitive,
    dim: usize,
    k: usize,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let vals = decode_ints(tensor.buffer().as_slice(), tensor.dtype());
    let mut out_dims = dims.to_vec();
    out_dims[dim] = k;
    let outer: usize = dims[..dim].iter().product();
    let dim_len = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    let mut out = vec![0i64; outer * k * inner];

    for o in 0..outer {
        for i in 0..inner {
            let mut pairs: Vec<(usize, i64)> = (0..dim_len)
                .map(|d| (d, vals[(o * dim_len + d) * inner + i]))
                .collect();
            pairs.sort_by_key(|a| std::cmp::Reverse(a.1));
            for (rank, (idx, _)) in pairs.into_iter().take(k).enumerate() {
                out[(o * k + rank) * inner + i] = idx as i64;
            }
        }
    }
    GnaIntTensorPrimitive::new(
        Shape::from(out_dims),
        IntDType::I32,
        GnaBuffer::from_bytes(encode_ints(&out, IntDType::I32)),
    )
}

pub fn execute_float_argtopk(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
    k: usize,
    out_dtype: IntDType,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let vals = decode_floats(tensor.buffer().as_slice(), tensor.dtype());
    let mut out_dims = dims.to_vec();
    out_dims[dim] = k;
    let outer: usize = dims[..dim].iter().product();
    let dim_len = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    let mut out = vec![0i64; outer * k * inner];

    for o in 0..outer {
        for i in 0..inner {
            let mut pairs: Vec<(usize, f32)> = (0..dim_len)
                .map(|d| (d, vals[(o * dim_len + d) * inner + i]))
                .collect();
            pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (rank, (idx, _)) in pairs.into_iter().take(k).enumerate() {
                out[(o * k + rank) * inner + i] = idx as i64;
            }
        }
    }
    GnaIntTensorPrimitive::new(
        Shape::from(out_dims),
        out_dtype,
        GnaBuffer::from_bytes(encode_ints(&out, out_dtype)),
    )
}

pub fn execute_int_cum(
    tensor: GnaIntTensorPrimitive,
    dim: usize,
    mode: CumMode,
) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let dtype = tensor.dtype();
    let vals = decode_ints(tensor.buffer().as_slice(), dtype);
    let outer: usize = dims[..dim].iter().product();
    let dim_len = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    let mut out = vec![0i64; vals.len()];

    for o in 0..outer {
        for i in 0..inner {
            let mut acc = vals[o * dim_len * inner + i];
            out[o * dim_len * inner + i] = acc;
            for d in 1..dim_len {
                let idx = (o * dim_len + d) * inner + i;
                let v = vals[idx];
                acc = match mode {
                    CumMode::Sum => acc.wrapping_add(v),
                    CumMode::Prod => acc.wrapping_mul(v),
                    CumMode::Min => acc.min(v),
                    CumMode::Max => acc.max(v),
                };
                out[idx] = acc;
            }
        }
    }
    GnaIntTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_ints(&out, dtype)),
    )
}

pub fn execute_float_cum(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
    mode: CumMode,
) -> GnaFloatTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let dtype = tensor.dtype();
    let vals = decode_floats(tensor.buffer().as_slice(), dtype);
    let outer: usize = dims[..dim].iter().product();
    let dim_len = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    let mut out = vec![0.0f32; vals.len()];

    for o in 0..outer {
        for i in 0..inner {
            let mut acc = vals[o * dim_len * inner + i];
            out[o * dim_len * inner + i] = acc;
            for d in 1..dim_len {
                let idx = (o * dim_len + d) * inner + i;
                let v = vals[idx];
                acc = match mode {
                    CumMode::Sum => acc + v,
                    CumMode::Prod => acc * v,
                    CumMode::Min => acc.min(v),
                    CumMode::Max => acc.max(v),
                };
                out[idx] = acc;
            }
        }
    }
    GnaFloatTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub fn execute_int_prod(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
    let dtype = tensor.dtype();
    let prod: i64 = decode_ints(tensor.buffer().as_slice(), dtype)
        .into_iter()
        .product();
    GnaIntTensorPrimitive::new(
        Shape::from([1]),
        dtype,
        GnaBuffer::from_bytes(encode_ints(&[prod], dtype)),
    )
}

pub fn execute_int_prod_dim(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
    let shape = tensor.shape();
    let dims: &[usize] = &shape;
    let dtype = tensor.dtype();
    let vals = decode_ints(tensor.buffer().as_slice(), dtype);
    let (out_shape, outer, dim_len, inner) = reduce_dim_shape(dims, dim);
    let mut out = Vec::with_capacity(outer * inner);

    for o in 0..outer {
        for i in 0..inner {
            let p: i64 = (0..dim_len)
                .map(|d| vals[(o * dim_len + d) * inner + i])
                .product();
            out.push(p);
        }
    }
    GnaIntTensorPrimitive::new(
        out_shape,
        dtype,
        GnaBuffer::from_bytes(encode_ints(&out, dtype)),
    )
}

/// Execute 2D average pooling on Intel GNA 3.0 hardware (`gna_matmul_i16` window summation).
pub fn execute_float_avg_pool2d(
    x: GnaFloatTensorPrimitive,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> GnaFloatTensorPrimitive {
    let x_shape = x.shape();
    let xd: &[usize] = &x_shape;
    let (batch, channels, in_h, in_w) = (xd[0], xd[1], xd[2], xd[3]);
    let [kh, kw] = kernel_size;
    let [sh, sw] = stride;
    let [ph, pw] = padding;

    let out_h = if ceil_mode {
        (in_h + 2 * ph - kh).div_ceil(sh) + 1
    } else {
        (in_h + 2 * ph - kh) / sh + 1
    };
    let out_w = if ceil_mode {
        (in_w + 2 * pw - kw).div_ceil(sw) + 1
    } else {
        (in_w + 2 * pw - kw) / sw + 1
    };

    let dtype = x.dtype();
    let vals = decode_floats(x.buffer().as_slice(), dtype);
    let k_win = kh * kw;
    let m = batch * channels * out_h * out_w;

    let max_abs = vals.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let q_max = 16384.0f32;
    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / q_max };

    let mut col_i16 = vec![0i16; m * k_win];
    let mut counts = vec![0usize; m];

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let row = ((b * channels + c) * out_h + oh) * out_w + ow;
                    let mut valid = 0usize;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = (oh * sh + ky) as isize - ph as isize;
                            let ix = (ow * sw + kx) as isize - pw as isize;
                            let v =
                                if iy >= 0 && iy < in_h as isize && ix >= 0 && ix < in_w as isize {
                                    valid += 1;
                                    vals[((b * channels + c) * in_h + iy as usize) * in_w
                                        + ix as usize]
                                } else {
                                    0.0
                                };
                            col_i16[row * k_win + ky * kw + kx] =
                                (v / scale).round().clamp(-q_max, q_max) as i16;
                        }
                    }
                    counts[row] = if count_include_pad {
                        k_win
                    } else {
                        valid.max(1)
                    };
                }
            }
        }
    }

    let ones = vec![1i16; k_win];
    let sums = gna_matmul_i16(&col_i16, &ones, m, k_win, 1, 0)
        .expect("GNA hardware avg_pool2d execution failed");
    let out: Vec<f32> = sums
        .into_iter()
        .zip(counts)
        .map(|(s, cnt)| (s as f32 * scale) / cnt as f32)
        .collect();

    GnaFloatTensorPrimitive::new(
        Shape::from([batch, channels, out_h, out_w]),
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}

pub fn execute_float_adaptive_avg_pool2d(
    x: GnaFloatTensorPrimitive,
    output_size: [usize; 2],
) -> GnaFloatTensorPrimitive {
    let x_shape = x.shape();
    let xd: &[usize] = &x_shape;
    let (batch, channels, in_h, in_w) = (xd[0], xd[1], xd[2], xd[3]);
    let [out_h, out_w] = output_size;
    let dtype = x.dtype();
    let vals = decode_floats(x.buffer().as_slice(), dtype);
    let mut out = vec![0.0f32; batch * channels * out_h * out_w];

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                let h_start = (oh * in_h) / out_h;
                let h_end = ((oh + 1) * in_h).div_ceil(out_h);
                for ow in 0..out_w {
                    let w_start = (ow * in_w) / out_w;
                    let w_end = ((ow + 1) * in_w).div_ceil(out_w);
                    let mut sum = 0.0f32;
                    let mut cnt = 0usize;
                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            sum += vals[((b * channels + c) * in_h + ih) * in_w + iw];
                            cnt += 1;
                        }
                    }
                    out[((b * channels + c) * out_h + oh) * out_w + ow] = sum / cnt.max(1) as f32;
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

pub fn execute_float_max_pool2d_with_indices(
    x: GnaFloatTensorPrimitive,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    indices_dtype: IntDType,
) -> (GnaFloatTensorPrimitive, GnaIntTensorPrimitive) {
    let x_shape = x.shape();
    let xd: &[usize] = &x_shape;
    let (batch, channels, in_h, in_w) = (xd[0], xd[1], xd[2], xd[3]);
    let [kh, kw] = kernel_size;
    let [sh, sw] = stride;
    let [ph, pw] = padding;
    let [dh, dw] = dilation;

    let out_h = if ceil_mode {
        (in_h + 2 * ph - dh * (kh - 1) - 1).div_ceil(sh) + 1
    } else {
        (in_h + 2 * ph - dh * (kh - 1) - 1) / sh + 1
    };
    let out_w = if ceil_mode {
        (in_w + 2 * pw - dw * (kw - 1) - 1).div_ceil(sw) + 1
    } else {
        (in_w + 2 * pw - dw * (kw - 1) - 1) / sw + 1
    };

    let dtype = x.dtype();
    let vals = decode_floats(x.buffer().as_slice(), dtype);
    let mut out = vec![0.0f32; batch * channels * out_h * out_w];
    let mut indices = vec![0i64; batch * channels * out_h * out_w];

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut best = f32::NEG_INFINITY;
                    let mut best_idx = 0i64;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = (oh * sh + ky * dh) as isize - ph as isize;
                            let ix = (ow * sw + kx * dw) as isize - pw as isize;
                            if iy >= 0 && iy < in_h as isize && ix >= 0 && ix < in_w as isize {
                                let lin = (iy as usize) * in_w + (ix as usize);
                                let v = vals[(b * channels + c) * in_h * in_w + lin];
                                if v > best {
                                    best = v;
                                    best_idx = lin as i64;
                                }
                            }
                        }
                    }
                    let out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                    out[out_idx] = best;
                    indices[out_idx] = best_idx;
                }
            }
        }
    }

    let out_shape = Shape::from([batch, channels, out_h, out_w]);
    (
        GnaFloatTensorPrimitive::new(
            out_shape.clone(),
            dtype,
            GnaBuffer::from_bytes(encode_floats(&out, dtype)),
        ),
        GnaIntTensorPrimitive::new(
            out_shape,
            indices_dtype,
            GnaBuffer::from_bytes(encode_ints(&indices, indices_dtype)),
        ),
    )
}

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
    let l = decode_floats(lhs.buffer().as_slice(), dtype);
    let r = decode_floats(rhs.buffer().as_slice(), rhs.dtype());
    let out: Vec<f32> = (0..l.len()).map(|i| f(l[i], r[i % r.len()])).collect();
    GnaFloatTensorPrimitive::new(
        lhs.shape(),
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
    let l = decode_ints(lhs.buffer().as_slice(), dtype);
    let r = decode_ints(rhs.buffer().as_slice(), rhs.dtype());
    let out: Vec<i64> = (0..l.len()).map(|i| f(l[i], r[i % r.len()])).collect();
    GnaIntTensorPrimitive::new(
        lhs.shape(),
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

static GNA_RNG_STATE: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);

pub fn set_rng_seed(seed: u64) {
    GNA_RNG_STATE.store(seed ^ 0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
}

fn next_u64() -> u64 {
    let mut prev = GNA_RNG_STATE.load(Ordering::Relaxed);
    loop {
        let mut z = prev.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let next_state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let out = z ^ (z >> 31);
        match GNA_RNG_STATE.compare_exchange_weak(
            prev,
            next_state,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return out,
            Err(actual) => prev = actual,
        }
    }
}

fn next_f64_unit() -> f64 {
    ((next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

fn sample_f64(distribution: Distribution) -> f64 {
    match distribution {
        Distribution::Default => next_f64_unit(),
        Distribution::Bernoulli(p) => {
            if next_f64_unit() < p {
                1.0
            } else {
                0.0
            }
        }
        Distribution::Uniform(low, high) => low + (high - low) * next_f64_unit(),
        Distribution::Normal(mean, std) => {
            let u1 = next_f64_unit().max(1e-12);
            let u2 = next_f64_unit();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            mean + std * z
        }
    }
}

pub fn execute_float_random(
    shape: Shape,
    distribution: Distribution,
    dtype: FloatDType,
) -> GnaFloatTensorPrimitive {
    let num = shape.num_elements();
    let vals: Vec<f32> = (0..num).map(|_| sample_f64(distribution) as f32).collect();
    GnaFloatTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_floats(&vals, dtype)),
    )
}

pub fn execute_int_random(
    shape: Shape,
    distribution: Distribution,
    dtype: IntDType,
) -> GnaIntTensorPrimitive {
    let num = shape.num_elements();
    let vals: Vec<i64> = (0..num)
        .map(|_| match distribution {
            Distribution::Default => (next_u64() % 100) as i64,
            _ => sample_f64(distribution).round() as i64,
        })
        .collect();
    GnaIntTensorPrimitive::new(
        shape,
        dtype,
        GnaBuffer::from_bytes(encode_ints(&vals, dtype)),
    )
}
