use crate::backend::gna_exec::{flash_attn, module_ops};
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::float::GnaFloatTensorPrimitive;
use burn_backend::ops::{AttentionModuleOptions, InterpolateOptions};

pub(super) fn interpolate(
    x: GnaFloatTensorPrimitive,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> GnaFloatTensorPrimitive {
    module_ops::execute_float_interpolate(x, output_size, options)
}

pub(super) fn interpolate_backward(
    x: GnaFloatTensorPrimitive,
    grad: GnaFloatTensorPrimitive,
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> GnaFloatTensorPrimitive {
    module_ops::execute_float_interpolate_backward(x, grad, output_size, options)
}

pub(super) fn attention(
    query: GnaFloatTensorPrimitive,
    key: GnaFloatTensorPrimitive,
    value: GnaFloatTensorPrimitive,
    mask: Option<GnaBoolTensorPrimitive>,
    attn_bias: Option<GnaFloatTensorPrimitive>,
    options: AttentionModuleOptions,
) -> GnaFloatTensorPrimitive {
    flash_attn::execute_flash_attention(query, key, value, mask, attn_bias, options)
}
