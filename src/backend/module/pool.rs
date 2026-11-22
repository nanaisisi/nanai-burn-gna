use crate::backend::GnaBackend;
use crate::backend::gna_exec::{extra, module_ops};
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::IntDType;
use burn_backend::ops::{MaxPool2dBackward, MaxPool2dWithIndices};

pub(super) fn avg_pool2d(
    x: GnaFloatTensorPrimitive,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> GnaFloatTensorPrimitive {
    extra::execute_float_avg_pool2d(
        x,
        extra::pool::AvgPool2dParams {
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        },
    )
}

pub(super) fn avg_pool2d_backward(
    x: GnaFloatTensorPrimitive,
    grad: GnaFloatTensorPrimitive,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> GnaFloatTensorPrimitive {
    module_ops::execute_float_avg_pool2d_backward(
        x,
        grad,
        kernel_size,
        stride,
        padding,
        count_include_pad,
        ceil_mode,
    )
}

pub(super) fn adaptive_avg_pool2d(
    x: GnaFloatTensorPrimitive,
    output_size: [usize; 2],
) -> GnaFloatTensorPrimitive {
    extra::execute_float_adaptive_avg_pool2d(x, output_size)
}

pub(super) fn adaptive_avg_pool2d_backward(
    x: GnaFloatTensorPrimitive,
    grad: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    module_ops::execute_float_adaptive_avg_pool2d_backward(x, grad)
}

pub(super) fn adaptive_avg_pool3d(
    x: GnaFloatTensorPrimitive,
    output_size: [usize; 3],
) -> GnaFloatTensorPrimitive {
    module_ops::execute_float_adaptive_avg_pool3d(x, output_size)
}

pub(super) fn adaptive_avg_pool3d_backward(
    x: GnaFloatTensorPrimitive,
    grad: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    module_ops::execute_float_adaptive_avg_pool3d_backward(x, grad)
}

pub(super) fn max_pool2d(
    x: GnaFloatTensorPrimitive,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> GnaFloatTensorPrimitive {
    extra::execute_float_max_pool2d_with_indices(
        x,
        extra::pool::MaxPool2dParams {
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        },
        IntDType::I32,
    )
    .0
}

pub(super) fn max_pool2d_with_indices(
    x: GnaFloatTensorPrimitive,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
    indices_dtype: IntDType,
) -> MaxPool2dWithIndices<GnaBackend> {
    let (out, indices) = extra::execute_float_max_pool2d_with_indices(
        x,
        extra::pool::MaxPool2dParams {
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        },
        indices_dtype,
    );
    MaxPool2dWithIndices::new(out, indices)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn max_pool2d_with_indices_backward(
    x: GnaFloatTensorPrimitive,
    _kernel_size: [usize; 2],
    _stride: [usize; 2],
    _padding: [usize; 2],
    _dilation: [usize; 2],
    _ceil_mode: bool,
    output_grad: GnaFloatTensorPrimitive,
    indices: GnaIntTensorPrimitive,
) -> MaxPool2dBackward<GnaBackend> {
    module_ops::execute_float_max_pool2d_backward(x, output_grad, indices)
}
