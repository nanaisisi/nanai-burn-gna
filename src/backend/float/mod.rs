use crate::backend::{GnaBackend, UnfoldParams};
use crate::device::GnaDevice;
use crate::tensor::{
    bool::GnaBoolTensorPrimitive, float::GnaFloatTensorPrimitive, int::GnaIntTensorPrimitive,
};
use burn_backend::ops::FloatTensorOps;
use burn_backend::tensor::IndexingUpdateOp;
use burn_backend::{
    BoolStore, Distribution, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
};

mod arithmetic;
mod comparison;
mod creation;
mod indexing;
mod reduce;
mod unary;

impl FloatTensorOps<Self> for GnaBackend {
    fn float_from_data(data: TensorData, device: &GnaDevice) -> GnaFloatTensorPrimitive {
        creation::float_from_data(data, device)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &GnaDevice,
        dtype: FloatDType,
    ) -> GnaFloatTensorPrimitive {
        creation::float_random(shape, distribution, device, dtype)
    }

    async fn float_into_data(
        tensor: GnaFloatTensorPrimitive,
    ) -> Result<TensorData, ExecutionError> {
        creation::float_into_data(tensor).await
    }

    fn float_to_device(
        tensor: GnaFloatTensorPrimitive,
        device: &GnaDevice,
    ) -> GnaFloatTensorPrimitive {
        creation::float_to_device(tensor, device)
    }

    fn float_into_int(
        tensor: GnaFloatTensorPrimitive,
        out_dtype: IntDType,
    ) -> GnaIntTensorPrimitive {
        creation::float_into_int(tensor, out_dtype)
    }

    fn float_empty(shape: Shape, device: &GnaDevice, dtype: FloatDType) -> GnaFloatTensorPrimitive {
        creation::float_empty(shape, device, dtype)
    }

    fn float_cast(tensor: GnaFloatTensorPrimitive, dtype: FloatDType) -> GnaFloatTensorPrimitive {
        creation::float_cast(tensor, dtype)
    }

    fn float_expand(tensor: GnaFloatTensorPrimitive, shape: Shape) -> GnaFloatTensorPrimitive {
        creation::float_expand(tensor, shape)
    }

    fn float_reshape(tensor: GnaFloatTensorPrimitive, shape: Shape) -> GnaFloatTensorPrimitive {
        creation::float_reshape(tensor, shape)
    }

    fn float_add(
        lhs: GnaFloatTensorPrimitive,
        rhs: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        arithmetic::float_add(lhs, rhs)
    }

    fn float_add_scalar(lhs: GnaFloatTensorPrimitive, rhs: Scalar) -> GnaFloatTensorPrimitive {
        arithmetic::float_add_scalar(lhs, rhs)
    }

    fn float_sub(
        lhs: GnaFloatTensorPrimitive,
        rhs: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        arithmetic::float_sub(lhs, rhs)
    }

    fn float_sub_scalar(lhs: GnaFloatTensorPrimitive, rhs: Scalar) -> GnaFloatTensorPrimitive {
        arithmetic::float_sub_scalar(lhs, rhs)
    }

    fn float_mul(
        lhs: GnaFloatTensorPrimitive,
        rhs: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        arithmetic::float_mul(lhs, rhs)
    }

    fn float_mul_scalar(lhs: GnaFloatTensorPrimitive, rhs: Scalar) -> GnaFloatTensorPrimitive {
        arithmetic::float_mul_scalar(lhs, rhs)
    }

    fn float_div(
        lhs: GnaFloatTensorPrimitive,
        rhs: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        arithmetic::float_div(lhs, rhs)
    }

    fn float_div_scalar(lhs: GnaFloatTensorPrimitive, rhs: Scalar) -> GnaFloatTensorPrimitive {
        arithmetic::float_div_scalar(lhs, rhs)
    }

    fn float_remainder(
        lhs: GnaFloatTensorPrimitive,
        rhs: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        arithmetic::float_remainder(lhs, rhs)
    }

    fn float_remainder_scalar(
        lhs: GnaFloatTensorPrimitive,
        rhs: Scalar,
    ) -> GnaFloatTensorPrimitive {
        arithmetic::float_remainder_scalar(lhs, rhs)
    }

    fn float_matmul(
        lhs: GnaFloatTensorPrimitive,
        rhs: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        arithmetic::float_matmul(lhs, rhs)
    }

    fn float_cross(
        lhs: GnaFloatTensorPrimitive,
        rhs: GnaFloatTensorPrimitive,
        dim: usize,
    ) -> GnaFloatTensorPrimitive {
        arithmetic::float_cross(lhs, rhs, dim)
    }

    fn float_recip(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        arithmetic::float_recip(tensor)
    }

    fn float_powf(
        lhs: GnaFloatTensorPrimitive,
        rhs: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        arithmetic::float_powf(lhs, rhs)
    }

    fn float_powf_scalar_impl(
        tensor: GnaFloatTensorPrimitive,
        value: Scalar,
    ) -> GnaFloatTensorPrimitive {
        arithmetic::float_powf_scalar_impl(tensor, value)
    }

    fn float_atan2(
        lhs: GnaFloatTensorPrimitive,
        rhs: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        arithmetic::float_atan2(lhs, rhs)
    }

    fn float_sum(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        reduce::float_sum(tensor)
    }

    fn float_sum_dim(tensor: GnaFloatTensorPrimitive, dim: usize) -> GnaFloatTensorPrimitive {
        reduce::float_sum_dim(tensor, dim)
    }

    fn float_mean_dim(tensor: GnaFloatTensorPrimitive, dim: usize) -> GnaFloatTensorPrimitive {
        reduce::float_mean_dim(tensor, dim)
    }

    fn float_cumsum(tensor: GnaFloatTensorPrimitive, dim: usize) -> GnaFloatTensorPrimitive {
        reduce::float_cumsum(tensor, dim)
    }

    fn float_cumprod(tensor: GnaFloatTensorPrimitive, dim: usize) -> GnaFloatTensorPrimitive {
        reduce::float_cumprod(tensor, dim)
    }

    fn float_cummin(tensor: GnaFloatTensorPrimitive, dim: usize) -> GnaFloatTensorPrimitive {
        reduce::float_cummin(tensor, dim)
    }

    fn float_cummax(tensor: GnaFloatTensorPrimitive, dim: usize) -> GnaFloatTensorPrimitive {
        reduce::float_cummax(tensor, dim)
    }

    fn float_exp(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_exp(tensor)
    }

    fn float_log(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_log(tensor)
    }

    fn float_log1p(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_log1p(tensor)
    }

    fn float_sqrt(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_sqrt(tensor)
    }

    fn float_abs(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_abs(tensor)
    }

    fn float_cos(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_cos(tensor)
    }

    fn float_sin(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_sin(tensor)
    }

    fn float_tan(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_tan(tensor)
    }

    fn float_cosh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_cosh(tensor)
    }

    fn float_sinh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_sinh(tensor)
    }

    fn float_tanh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_tanh(tensor)
    }

    fn float_acos(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_acos(tensor)
    }

    fn float_acosh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_acosh(tensor)
    }

    fn float_asin(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_asin(tensor)
    }

    fn float_asinh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_asinh(tensor)
    }

    fn float_atan(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_atan(tensor)
    }

    fn float_atanh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_atanh(tensor)
    }

    fn float_round(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_round(tensor)
    }

    fn float_floor(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_floor(tensor)
    }

    fn float_ceil(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_ceil(tensor)
    }

    fn float_trunc(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_trunc(tensor)
    }

    fn float_erf(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        unary::float_erf(tensor)
    }

    fn float_argmax(
        tensor: GnaFloatTensorPrimitive,
        dim: usize,
        out_dtype: IntDType,
    ) -> GnaIntTensorPrimitive {
        reduce::float_argmax(tensor, dim, out_dtype)
    }

    fn float_argtopk(
        tensor: GnaFloatTensorPrimitive,
        dim: usize,
        k: usize,
        out_dtype: IntDType,
    ) -> GnaIntTensorPrimitive {
        reduce::float_argtopk(tensor, dim, k, out_dtype)
    }

    fn float_argmin(
        tensor: GnaFloatTensorPrimitive,
        dim: usize,
        out_dtype: IntDType,
    ) -> GnaIntTensorPrimitive {
        reduce::float_argmin(tensor, dim, out_dtype)
    }

    fn float_unfold(
        tensor: GnaFloatTensorPrimitive,
        dim: usize,
        size: usize,
        step: usize,
    ) -> GnaFloatTensorPrimitive {
        indexing::float_unfold(tensor, UnfoldParams::new(dim, size, step))
    }

    fn float_equal(
        lhs: GnaFloatTensorPrimitive,
        rhs: GnaFloatTensorPrimitive,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::float_equal(lhs, rhs, out_dtype)
    }

    fn float_equal_elem(
        lhs: GnaFloatTensorPrimitive,
        rhs: Scalar,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::float_equal_elem(lhs, rhs, out_dtype)
    }

    fn float_greater(
        lhs: GnaFloatTensorPrimitive,
        rhs: GnaFloatTensorPrimitive,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::float_greater(lhs, rhs, out_dtype)
    }

    fn float_greater_elem(
        lhs: GnaFloatTensorPrimitive,
        rhs: Scalar,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::float_greater_elem(lhs, rhs, out_dtype)
    }

    fn float_greater_equal(
        lhs: GnaFloatTensorPrimitive,
        rhs: GnaFloatTensorPrimitive,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::float_greater_equal(lhs, rhs, out_dtype)
    }

    fn float_greater_equal_elem(
        lhs: GnaFloatTensorPrimitive,
        rhs: Scalar,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::float_greater_equal_elem(lhs, rhs, out_dtype)
    }

    fn float_lower(
        lhs: GnaFloatTensorPrimitive,
        rhs: GnaFloatTensorPrimitive,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::float_lower(lhs, rhs, out_dtype)
    }

    fn float_lower_elem(
        lhs: GnaFloatTensorPrimitive,
        rhs: Scalar,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::float_lower_elem(lhs, rhs, out_dtype)
    }

    fn float_lower_equal(
        lhs: GnaFloatTensorPrimitive,
        rhs: GnaFloatTensorPrimitive,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::float_lower_equal(lhs, rhs, out_dtype)
    }

    fn float_lower_equal_elem(
        lhs: GnaFloatTensorPrimitive,
        rhs: Scalar,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::float_lower_equal_elem(lhs, rhs, out_dtype)
    }

    fn float_swap_dims(
        tensor: GnaFloatTensorPrimitive,
        dim1: usize,
        dim2: usize,
    ) -> GnaFloatTensorPrimitive {
        indexing::float_swap_dims(tensor, dim1, dim2)
    }

    fn float_permute(tensor: GnaFloatTensorPrimitive, axes: &[usize]) -> GnaFloatTensorPrimitive {
        indexing::float_permute(tensor, axes)
    }

    fn float_flip(tensor: GnaFloatTensorPrimitive, axes: &[usize]) -> GnaFloatTensorPrimitive {
        indexing::float_flip(tensor, axes)
    }

    fn float_gather(
        dim: usize,
        tensor: GnaFloatTensorPrimitive,
        indices: GnaIntTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        indexing::float_gather(dim, tensor, indices)
    }

    fn float_scatter(
        dim: usize,
        tensor: GnaFloatTensorPrimitive,
        indices: GnaIntTensorPrimitive,
        value: GnaFloatTensorPrimitive,
        update: IndexingUpdateOp,
    ) -> GnaFloatTensorPrimitive {
        indexing::float_scatter(dim, tensor, indices, value, update)
    }

    fn float_select(
        tensor: GnaFloatTensorPrimitive,
        dim: usize,
        indices: GnaIntTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        indexing::float_select(tensor, dim, indices)
    }

    fn float_select_assign(
        tensor: GnaFloatTensorPrimitive,
        dim: usize,
        indices: GnaIntTensorPrimitive,
        value: GnaFloatTensorPrimitive,
        update: IndexingUpdateOp,
    ) -> GnaFloatTensorPrimitive {
        indexing::float_select_assign(tensor, dim, indices, value, update)
    }

    fn float_slice(tensor: GnaFloatTensorPrimitive, slices: &[Slice]) -> GnaFloatTensorPrimitive {
        indexing::float_slice(tensor, slices)
    }

    fn float_slice_assign(
        tensor: GnaFloatTensorPrimitive,
        slices: &[Slice],
        value: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        indexing::float_slice_assign(tensor, slices, value)
    }

    fn float_mask_where(
        tensor: GnaFloatTensorPrimitive,
        mask: GnaBoolTensorPrimitive,
        value: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        indexing::float_mask_where(tensor, mask, value)
    }

    fn float_mask_fill(
        tensor: GnaFloatTensorPrimitive,
        mask: GnaBoolTensorPrimitive,
        value: Scalar,
    ) -> GnaFloatTensorPrimitive {
        indexing::float_mask_fill(tensor, mask, value)
    }
}
