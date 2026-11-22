use std::sync::atomic::{AtomicU64, Ordering};

use burn_backend::{Distribution, FloatDType, IntDType, Shape};

use crate::backend::gna_exec::{encode_floats, encode_ints};
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;

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
