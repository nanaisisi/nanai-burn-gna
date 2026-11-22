use burn_backend::{IntDType, Shape, TensorMetadata};

use crate::backend::gna_exec::{
    GnaMatmulDims, decode_floats, encode_floats, encode_ints, gna_matmul_i16,
};
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;

/// Parameters for 2D average pooling.
#[derive(Clone, Copy, Debug)]
pub struct AvgPool2dParams {
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
}

impl AvgPool2dParams {
    pub const fn new(
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        }
    }

    pub const fn kernel_size(&self) -> [usize; 2] {
        self.kernel_size
    }

    pub const fn stride(&self) -> [usize; 2] {
        self.stride
    }

    pub const fn padding(&self) -> [usize; 2] {
        self.padding
    }

    pub const fn count_include_pad(&self) -> bool {
        self.count_include_pad
    }

    pub const fn ceil_mode(&self) -> bool {
        self.ceil_mode
    }
}

/// Parameters for 2D max pooling.
#[derive(Clone, Copy, Debug)]
pub struct MaxPool2dParams {
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
}

impl MaxPool2dParams {
    pub const fn new(
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        }
    }

    pub const fn kernel_size(&self) -> [usize; 2] {
        self.kernel_size
    }

    pub const fn stride(&self) -> [usize; 2] {
        self.stride
    }

    pub const fn padding(&self) -> [usize; 2] {
        self.padding
    }

    pub const fn dilation(&self) -> [usize; 2] {
        self.dilation
    }

    pub const fn ceil_mode(&self) -> bool {
        self.ceil_mode
    }
}

/// Execute 2D average pooling on Intel GNA 3.0 hardware (`gna_matmul_i16` window summation).
pub fn execute_float_avg_pool2d(
    x: GnaFloatTensorPrimitive,
    params: AvgPool2dParams,
) -> GnaFloatTensorPrimitive {
    let x_shape = x.shape();
    let xd: &[usize] = &x_shape;
    let (batch, channels, in_h, in_w) = (xd[0], xd[1], xd[2], xd[3]);
    let [kh, kw] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let count_include_pad = params.count_include_pad();
    let ceil_mode = params.ceil_mode();

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
    let sums = gna_matmul_i16(&col_i16, &ones, GnaMatmulDims::new(m, k_win, 1, 0))
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
    params: MaxPool2dParams,
    indices_dtype: IntDType,
) -> (GnaFloatTensorPrimitive, GnaIntTensorPrimitive) {
    let x_shape = x.shape();
    let xd: &[usize] = &x_shape;
    let (batch, channels, in_h, in_w) = (xd[0], xd[1], xd[2], xd[3]);
    let [kh, kw] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();
    let ceil_mode = params.ceil_mode();

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
