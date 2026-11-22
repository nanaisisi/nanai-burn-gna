use crate::backend::GnaBackend;
use crate::backend::gna_exec::{extra, module_ops};
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::IntDType;
use burn_backend::ops::{MaxPool2dBackward, MaxPool2dWithIndices};

pub(super) fn avg_pool2d(
    x: GnaFloatTensorPrimitive,
    params: extra::pool::AvgPool2dParams,
) -> GnaFloatTensorPrimitive {
    extra::execute_float_avg_pool2d(x, params)
}

pub(super) fn avg_pool2d_backward(
    x: GnaFloatTensorPrimitive,
    grad: GnaFloatTensorPrimitive,
    params: extra::pool::AvgPool2dParams,
) -> GnaFloatTensorPrimitive {
    module_ops::execute_float_avg_pool2d_backward(x, grad, params)
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
    params: extra::pool::MaxPool2dParams,
) -> GnaFloatTensorPrimitive {
    extra::execute_float_max_pool2d_with_indices(x, params, IntDType::I32).0
}

pub(super) fn max_pool2d_with_indices(
    x: GnaFloatTensorPrimitive,
    params: extra::pool::MaxPool2dParams,
    indices_dtype: IntDType,
) -> MaxPool2dWithIndices<GnaBackend> {
    let (out, indices) = extra::execute_float_max_pool2d_with_indices(x, params, indices_dtype);
    MaxPool2dWithIndices::new(out, indices)
}

pub(super) fn max_pool2d_with_indices_backward(
    x: GnaFloatTensorPrimitive,
    _params: extra::pool::MaxPool2dParams,
    output_grad: GnaFloatTensorPrimitive,
    indices: GnaIntTensorPrimitive,
) -> MaxPool2dBackward<GnaBackend> {
    module_ops::execute_float_max_pool2d_backward(x, output_grad, indices)
}
