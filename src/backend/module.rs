use super::GnaBackend;
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::ops::{
    AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConv2dBackward,
    DeformConvOptions, InterpolateOptions, MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
};
use burn_backend::{IntDType, TensorMetadata};

// ---------------------------------------------------------------------------
// ModuleOps
// ---------------------------------------------------------------------------
impl ModuleOps<Self> for GnaBackend {
    fn conv2d(
        x: GnaFloatTensorPrimitive,
        _weight: GnaFloatTensorPrimitive,
        _bias: Option<GnaFloatTensorPrimitive>,
        _options: ConvOptions<2>,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn deform_conv2d(
        x: GnaFloatTensorPrimitive,
        _offset: GnaFloatTensorPrimitive,
        _weight: GnaFloatTensorPrimitive,
        _mask: Option<GnaFloatTensorPrimitive>,
        _bias: Option<GnaFloatTensorPrimitive>,
        _options: DeformConvOptions<2>,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn deform_conv2d_backward(
        _x: GnaFloatTensorPrimitive,
        _offset: GnaFloatTensorPrimitive,
        _weight: GnaFloatTensorPrimitive,
        _mask: Option<GnaFloatTensorPrimitive>,
        _bias: Option<GnaFloatTensorPrimitive>,
        _output_grad: GnaFloatTensorPrimitive,
        _options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        unimplemented!("backward pass is not supported on GNA inference accelerator")
    }

    fn conv3d(
        x: GnaFloatTensorPrimitive,
        _weight: GnaFloatTensorPrimitive,
        _bias: Option<GnaFloatTensorPrimitive>,
        _options: ConvOptions<3>,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn conv_transpose2d(
        x: GnaFloatTensorPrimitive,
        _weight: GnaFloatTensorPrimitive,
        _bias: Option<GnaFloatTensorPrimitive>,
        _options: ConvTransposeOptions<2>,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn conv_transpose3d(
        x: GnaFloatTensorPrimitive,
        _weight: GnaFloatTensorPrimitive,
        _bias: Option<GnaFloatTensorPrimitive>,
        _options: ConvTransposeOptions<3>,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn avg_pool2d(
        x: GnaFloatTensorPrimitive,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _count_include_pad: bool,
        _ceil_mode: bool,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn avg_pool2d_backward(
        _x: GnaFloatTensorPrimitive,
        _grad: GnaFloatTensorPrimitive,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _count_include_pad: bool,
        _ceil_mode: bool,
    ) -> GnaFloatTensorPrimitive {
        unimplemented!("backward pass is not supported on GNA inference accelerator")
    }

    fn adaptive_avg_pool2d(
        x: GnaFloatTensorPrimitive,
        _output_size: [usize; 2],
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn adaptive_avg_pool2d_backward(
        _x: GnaFloatTensorPrimitive,
        _grad: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        unimplemented!("backward pass is not supported on GNA inference accelerator")
    }

    fn adaptive_avg_pool3d(
        x: GnaFloatTensorPrimitive,
        _output_size: [usize; 3],
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn adaptive_avg_pool3d_backward(
        _x: GnaFloatTensorPrimitive,
        _grad: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        unimplemented!("backward pass is not supported on GNA inference accelerator")
    }

    fn max_pool2d(
        x: GnaFloatTensorPrimitive,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn max_pool2d_with_indices(
        x: GnaFloatTensorPrimitive,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
        indices_dtype: IntDType,
    ) -> MaxPool2dWithIndices<Self> {
        let indices = GnaIntTensorPrimitive::new(x.shape(), indices_dtype, x.buffer().clone());
        MaxPool2dWithIndices::new(x, indices)
    }

    fn max_pool2d_with_indices_backward(
        _x: GnaFloatTensorPrimitive,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
        _output_grad: GnaFloatTensorPrimitive,
        _indices: GnaIntTensorPrimitive,
    ) -> MaxPool2dBackward<Self> {
        unimplemented!("backward pass is not supported on GNA inference accelerator")
    }

    fn interpolate(
        x: GnaFloatTensorPrimitive,
        _output_size: [usize; 2],
        _options: InterpolateOptions,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn interpolate_backward(
        _x: GnaFloatTensorPrimitive,
        _grad: GnaFloatTensorPrimitive,
        _output_size: [usize; 2],
        _options: InterpolateOptions,
    ) -> GnaFloatTensorPrimitive {
        unimplemented!("backward pass is not supported on GNA inference accelerator")
    }

    fn attention(
        query: GnaFloatTensorPrimitive,
        _key: GnaFloatTensorPrimitive,
        _value: GnaFloatTensorPrimitive,
        _mask: Option<GnaBoolTensorPrimitive>,
        _attn_bias: Option<GnaFloatTensorPrimitive>,
        _options: AttentionModuleOptions,
    ) -> GnaFloatTensorPrimitive {
        query
    }
}
