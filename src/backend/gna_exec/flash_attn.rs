use burn_backend::ops::AttentionModuleOptions;
use burn_backend::{Shape, TensorMetadata};

use super::{decode_floats, encode_floats, execute_float_matmul};
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;

/// Tiled FlashAttention (Online Softmax) using Intel GNA 3.0 hardware `matmul`.
pub fn execute_flash_attention(
    query: GnaFloatTensorPrimitive,
    key: GnaFloatTensorPrimitive,
    value: GnaFloatTensorPrimitive,
    mask: Option<GnaBoolTensorPrimitive>,
    attn_bias: Option<GnaFloatTensorPrimitive>,
    options: AttentionModuleOptions,
) -> GnaFloatTensorPrimitive {
    const TILE_Q: usize = 64;
    const TILE_KV: usize = 64;

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

    let q_vals = decode_floats(query.buffer().as_slice(), dtype);
    let k_vals = decode_floats(key.buffer().as_slice(), key.dtype());
    let v_vals = decode_floats(value.buffer().as_slice(), value.dtype());
    let mask_bytes = mask.as_ref().map(|m| m.buffer().as_slice());
    let bias_vals = attn_bias
        .as_ref()
        .map(|b| decode_floats(b.buffer().as_slice(), b.dtype()));

    let mut out = vec![0.0f32; batch * h_q * s_q * d_v];
    let head_group = (h_q / h_kv).max(1);

    for b in 0..batch {
        for h in 0..h_q {
            let kv_h = h / head_group;
            let q_head_off = (b * h_q + h) * s_q * d_k;
            let k_head_off = (b * h_kv + kv_h) * s_k * d_k;
            let v_head_off = (b * h_kv + kv_h) * s_k * d_v;
            let out_head_off = (b * h_q + h) * s_q * d_v;

            for q_start in (0..s_q).step_by(TILE_Q) {
                let q_len = (s_q - q_start).min(TILE_Q);
                let mut m_i = vec![f32::NEG_INFINITY; q_len];
                let mut l_i = vec![0.0f32; q_len];
                let mut o_acc = vec![0.0f32; q_len * d_v];

                let q_tile_slice =
                    &q_vals[q_head_off + q_start * d_k..q_head_off + (q_start + q_len) * d_k];
                let q_tile_prim = GnaFloatTensorPrimitive::new(
                    Shape::from([q_len, d_k]),
                    dtype,
                    GnaBuffer::from_bytes(encode_floats(q_tile_slice, dtype)),
                );

                for kv_start in (0..s_k).step_by(TILE_KV) {
                    if options.is_causal && kv_start >= q_start + q_len {
                        break;
                    }
                    let kv_len = (s_k - kv_start).min(TILE_KV);

                    let mut kt_tile = vec![0.0f32; d_k * kv_len];
                    for r in 0..kv_len {
                        for c in 0..d_k {
                            kt_tile[c * kv_len + r] =
                                k_vals[k_head_off + (kv_start + r) * d_k + c];
                        }
                    }
                    let kt_prim = GnaFloatTensorPrimitive::new(
                        Shape::from([d_k, kv_len]),
                        dtype,
                        GnaBuffer::from_bytes(encode_floats(&kt_tile, dtype)),
                    );

                    // S_tile = Q_tile * K_tile^T on GNA 3.0 hardware
                    let s_prim = execute_float_matmul(q_tile_prim.clone(), kt_prim);
                    let mut s_tile = decode_floats(s_prim.buffer().as_slice(), dtype);

                    let mut p_tile = vec![0.0f32; q_len * kv_len];
                    let mut alphas = vec![1.0f32; q_len];

                    for r in 0..q_len {
                        let global_q = q_start + r;
                        let mut row_max = f32::NEG_INFINITY;
                        for c in 0..kv_len {
                            let global_k = kv_start + c;
                            let idx = r * kv_len + c;
                            let mut v = s_tile[idx] * scale;
                            if let Some(cap) = options.softcap {
                                let cap_f = cap as f32;
                                v = cap_f * (v / cap_f).tanh();
                            }
                            if let Some(ref bv) = bias_vals {
                                let g_idx = ((b * h_q + h) * s_q + global_q) * s_k + global_k;
                                v += bv[g_idx % bv.len()];
                            }
                            if options.is_causal && global_k > global_q {
                                v = f32::NEG_INFINITY;
                            }
                            if let Some(mb) = mask_bytes {
                                let g_idx = ((b * h_q + h) * s_q + global_q) * s_k + global_k;
                                if mb[g_idx % mb.len()] != 0 {
                                    v = f32::NEG_INFINITY;
                                }
                            }
                            s_tile[idx] = v;
                            if v > row_max {
                                row_max = v;
                            }
                        }

                        let m_new = m_i[r].max(row_max);
                        let alpha = if m_i[r].is_finite() {
                            (m_i[r] - m_new).exp()
                        } else {
                            0.0
                        };
                        alphas[r] = alpha;

                        let mut row_sum = 0.0f32;
                        for c in 0..kv_len {
                            let e = (s_tile[r * kv_len + c] - m_new).exp();
                            p_tile[r * kv_len + c] = e;
                            row_sum += e;
                        }
                        m_i[r] = m_new;
                        l_i[r] = alpha * l_i[r] + row_sum;
                    }

                    // PV_tile = P_tile * V_tile on GNA 3.0 hardware
                    let p_prim = GnaFloatTensorPrimitive::new(
                        Shape::from([q_len, kv_len]),
                        dtype,
                        GnaBuffer::from_bytes(encode_floats(&p_tile, dtype)),
                    );
                    let v_tile_slice = &v_vals
                        [v_head_off + kv_start * d_v..v_head_off + (kv_start + kv_len) * d_v];
                    let v_prim = GnaFloatTensorPrimitive::new(
                        Shape::from([kv_len, d_v]),
                        dtype,
                        GnaBuffer::from_bytes(encode_floats(v_tile_slice, dtype)),
                    );
                    let pv_prim = execute_float_matmul(p_prim, v_prim);
                    let pv_tile = decode_floats(pv_prim.buffer().as_slice(), dtype);

                    for r in 0..q_len {
                        let alpha = alphas[r];
                        for d in 0..d_v {
                            o_acc[r * d_v + d] =
                                o_acc[r * d_v + d] * alpha + pv_tile[r * d_v + d];
                        }
                    }
                }

                for r in 0..q_len {
                    let inv_l = if l_i[r] == 0.0 { 0.0 } else { 1.0 / l_i[r] };
                    for d in 0..d_v {
                        out[out_head_off + (q_start + r) * d_v + d] = o_acc[r * d_v + d] * inv_l;
                    }
                }
            }
        }
    }

    GnaFloatTensorPrimitive::new(
        Shape::from([batch, h_q, s_q, d_v]),
        dtype,
        GnaBuffer::from_bytes(encode_floats(&out, dtype)),
    )
}
