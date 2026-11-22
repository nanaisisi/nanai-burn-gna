use super::GnaBackend;
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::IntDType;
use burn_backend::ops::{
    AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConv2dBackward,
    DeformConvOptions, InterpolateOptions, MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
};

// ---------------------------------------------------------------------------
// ModuleOps
// ---------------------------------------------------------------------------
impl ModuleOps<Self> for GnaBackend {
    fn conv2d(
        x: GnaFloatTensorPrimitive,
        weight: GnaFloatTensorPrimitive,
        bias: Option<GnaFloatTensorPrimitive>,
        options: ConvOptions<2>,
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::ops::execute_float_conv2d(x, weight, bias, options)
    }

    fn deform_conv2d(
        x: GnaFloatTensorPrimitive,
        offset: GnaFloatTensorPrimitive,
        weight: GnaFloatTensorPrimitive,
        mask: Option<GnaFloatTensorPrimitive>,
        bias: Option<GnaFloatTensorPrimitive>,
        options: DeformConvOptions<2>,
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::module_ops::execute_float_deform_conv2d(
            x, offset, weight, mask, bias, options,
        )
    }

    fn deform_conv2d_backward(
        x: GnaFloatTensorPrimitive,
        offset: GnaFloatTensorPrimitive,
        weight: GnaFloatTensorPrimitive,
        mask: Option<GnaFloatTensorPrimitive>,
        bias: Option<GnaFloatTensorPrimitive>,
        output_grad: GnaFloatTensorPrimitive,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        crate::backend::gna_exec::module_ops::execute_float_deform_conv2d_backward(
            x,
            offset,
            weight,
            mask,
            bias,
            output_grad,
            options,
        )
    }

    fn conv3d(
        x: GnaFloatTensorPrimitive,
        weight: GnaFloatTensorPrimitive,
        bias: Option<GnaFloatTensorPrimitive>,
        options: ConvOptions<3>,
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::module_ops::execute_float_conv3d(x, weight, bias, options)
    }

    fn conv_transpose2d(
        x: GnaFloatTensorPrimitive,
        weight: GnaFloatTensorPrimitive,
        bias: Option<GnaFloatTensorPrimitive>,
        options: ConvTransposeOptions<2>,
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::module_ops::execute_float_conv_transpose2d(
            x, weight, bias, options,
        )
    }

    fn conv_transpose3d(
        x: GnaFloatTensorPrimitive,
        weight: GnaFloatTensorPrimitive,
        bias: Option<GnaFloatTensorPrimitive>,
        options: ConvTransposeOptions<3>,
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::module_ops::execute_float_conv_transpose3d(
            x, weight, bias, options,
        )
    }

    fn avg_pool2d(
        x: GnaFloatTensorPrimitive,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::extra::execute_float_avg_pool2d(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }

    fn avg_pool2d_backward(
        x: GnaFloatTensorPrimitive,
        grad: GnaFloatTensorPrimitive,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::module_ops::execute_float_avg_pool2d_backward(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        )
    }

    fn adaptive_avg_pool2d(
        x: GnaFloatTensorPrimitive,
        output_size: [usize; 2],
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::extra::execute_float_adaptive_avg_pool2d(x, output_size)
    }

    fn adaptive_avg_pool2d_backward(
        x: GnaFloatTensorPrimitive,
        grad: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::module_ops::execute_float_adaptive_avg_pool2d_backward(x, grad)
    }

    fn adaptive_avg_pool3d(
        x: GnaFloatTensorPrimitive,
        output_size: [usize; 3],
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::module_ops::execute_float_adaptive_avg_pool3d(x, output_size)
    }

    fn adaptive_avg_pool3d_backward(
        x: GnaFloatTensorPrimitive,
        grad: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::module_ops::execute_float_adaptive_avg_pool3d_backward(x, grad)
    }

    fn max_pool2d(
        x: GnaFloatTensorPrimitive,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::extra::execute_float_max_pool2d_with_indices(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            IntDType::I32,
        )
        .0
    }

    fn max_pool2d_with_indices(
        x: GnaFloatTensorPrimitive,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        indices_dtype: IntDType,
    ) -> MaxPool2dWithIndices<Self> {
        let (out, indices) = crate::backend::gna_exec::extra::execute_float_max_pool2d_with_indices(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            indices_dtype,
        );
        MaxPool2dWithIndices::new(out, indices)
    }

    fn max_pool2d_with_indices_backward(
        x: GnaFloatTensorPrimitive,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
        output_grad: GnaFloatTensorPrimitive,
        indices: GnaIntTensorPrimitive,
    ) -> MaxPool2dBackward<Self> {
        crate::backend::gna_exec::module_ops::execute_float_max_pool2d_backward(
            x,
            output_grad,
            indices,
        )
    }

    fn interpolate(
        x: GnaFloatTensorPrimitive,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::module_ops::execute_float_interpolate(x, output_size, options)
    }

    fn interpolate_backward(
        x: GnaFloatTensorPrimitive,
        grad: GnaFloatTensorPrimitive,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::module_ops::execute_float_interpolate_backward(
            x,
            grad,
            output_size,
            options,
        )
    }

    fn attention(
        query: GnaFloatTensorPrimitive,
        key: GnaFloatTensorPrimitive,
        value: GnaFloatTensorPrimitive,
        mask: Option<GnaBoolTensorPrimitive>,
        attn_bias: Option<GnaFloatTensorPrimitive>,
        options: AttentionModuleOptions,
    ) -> GnaFloatTensorPrimitive {
        crate::backend::gna_exec::flash_attn::execute_flash_attention(
            query, key, value, mask, attn_bias, options,
        )
    }
}
