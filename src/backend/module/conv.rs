use crate::backend::GnaBackend;
use crate::backend::gna_exec::{module_ops, ops};
use crate::tensor::float::GnaFloatTensorPrimitive;
use burn_backend::ops::{
    ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions,
};

pub(super) fn conv2d(
    x: GnaFloatTensorPrimitive,
    weight: GnaFloatTensorPrimitive,
    bias: Option<GnaFloatTensorPrimitive>,
    options: ConvOptions<2>,
) -> GnaFloatTensorPrimitive {
    ops::execute_float_conv2d(x, weight, bias, options)
}

pub(super) fn deform_conv2d(
    x: GnaFloatTensorPrimitive,
    offset: GnaFloatTensorPrimitive,
    weight: GnaFloatTensorPrimitive,
    mask: Option<GnaFloatTensorPrimitive>,
    bias: Option<GnaFloatTensorPrimitive>,
    options: DeformConvOptions<2>,
) -> GnaFloatTensorPrimitive {
    module_ops::execute_float_deform_conv2d(x, offset, weight, mask, bias, options)
}

pub(super) fn deform_conv2d_backward(
    x: GnaFloatTensorPrimitive,
    offset: GnaFloatTensorPrimitive,
    weight: GnaFloatTensorPrimitive,
    mask: Option<GnaFloatTensorPrimitive>,
    bias: Option<GnaFloatTensorPrimitive>,
    output_grad: GnaFloatTensorPrimitive,
    options: DeformConvOptions<2>,
) -> DeformConv2dBackward<GnaBackend> {
    module_ops::execute_float_deform_conv2d_backward(
        x,
        offset,
        weight,
        mask,
        bias,
        output_grad,
        options,
    )
}

pub(super) fn conv3d(
    x: GnaFloatTensorPrimitive,
    weight: GnaFloatTensorPrimitive,
    bias: Option<GnaFloatTensorPrimitive>,
    options: ConvOptions<3>,
) -> GnaFloatTensorPrimitive {
    module_ops::execute_float_conv3d(x, weight, bias, options)
}

pub(super) fn conv_transpose2d(
    x: GnaFloatTensorPrimitive,
    weight: GnaFloatTensorPrimitive,
    bias: Option<GnaFloatTensorPrimitive>,
    options: ConvTransposeOptions<2>,
) -> GnaFloatTensorPrimitive {
    module_ops::execute_float_conv_transpose2d(x, weight, bias, options)
}

pub(super) fn conv_transpose3d(
    x: GnaFloatTensorPrimitive,
    weight: GnaFloatTensorPrimitive,
    bias: Option<GnaFloatTensorPrimitive>,
    options: ConvTransposeOptions<3>,
) -> GnaFloatTensorPrimitive {
    module_ops::execute_float_conv_transpose3d(x, weight, bias, options)
}
