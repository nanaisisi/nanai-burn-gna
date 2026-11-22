use crate::backend::{GnaBackend, UnfoldParams};
use crate::device::GnaDevice;
use crate::tensor::{
    bool::GnaBoolTensorPrimitive, float::GnaFloatTensorPrimitive, int::GnaIntTensorPrimitive,
};
use burn_backend::ops::IntTensorOps;
use burn_backend::tensor::IndexingUpdateOp;
use burn_backend::{
    BoolStore, Distribution, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
};

mod arithmetic;
mod bitwise;
mod comparison;
mod creation;
mod indexing;
mod reduce;
mod unary;

impl IntTensorOps<Self> for GnaBackend {
    fn int_empty(shape: Shape, device: &GnaDevice, dtype: IntDType) -> GnaIntTensorPrimitive {
        creation::int_empty(shape, device, dtype)
    }

    async fn int_into_data(tensor: GnaIntTensorPrimitive) -> Result<TensorData, ExecutionError> {
        creation::int_into_data(tensor).await
    }

    fn int_from_data(data: TensorData, device: &GnaDevice) -> GnaIntTensorPrimitive {
        creation::int_from_data(data, device)
    }

    fn int_to_device(tensor: GnaIntTensorPrimitive, device: &GnaDevice) -> GnaIntTensorPrimitive {
        creation::int_to_device(tensor, device)
    }

    fn int_reshape(tensor: GnaIntTensorPrimitive, shape: Shape) -> GnaIntTensorPrimitive {
        creation::int_reshape(tensor, shape)
    }

    fn int_into_float(
        tensor: GnaIntTensorPrimitive,
        out_dtype: FloatDType,
    ) -> GnaFloatTensorPrimitive {
        creation::int_into_float(tensor, out_dtype)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &GnaDevice,
        dtype: IntDType,
    ) -> GnaIntTensorPrimitive {
        creation::int_random(shape, distribution, device, dtype)
    }

    fn int_expand(tensor: GnaIntTensorPrimitive, shape: Shape) -> GnaIntTensorPrimitive {
        creation::int_expand(tensor, shape)
    }

    fn int_cast(tensor: GnaIntTensorPrimitive, dtype: IntDType) -> GnaIntTensorPrimitive {
        creation::int_cast(tensor, dtype)
    }

    fn int_add(lhs: GnaIntTensorPrimitive, rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        arithmetic::int_add(lhs, rhs)
    }

    fn int_add_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
        arithmetic::int_add_scalar(lhs, rhs)
    }

    fn int_sub(lhs: GnaIntTensorPrimitive, rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        arithmetic::int_sub(lhs, rhs)
    }

    fn int_sub_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
        arithmetic::int_sub_scalar(lhs, rhs)
    }

    fn int_mul(lhs: GnaIntTensorPrimitive, rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        arithmetic::int_mul(lhs, rhs)
    }

    fn int_mul_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
        arithmetic::int_mul_scalar(lhs, rhs)
    }

    fn int_div(lhs: GnaIntTensorPrimitive, rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        arithmetic::int_div(lhs, rhs)
    }

    fn int_div_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
        arithmetic::int_div_scalar(lhs, rhs)
    }

    fn int_remainder(
        lhs: GnaIntTensorPrimitive,
        rhs: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        arithmetic::int_remainder(lhs, rhs)
    }

    fn int_remainder_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
        arithmetic::int_remainder_scalar(lhs, rhs)
    }

    fn int_matmul(lhs: GnaIntTensorPrimitive, rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        arithmetic::int_matmul(lhs, rhs)
    }

    fn bitwise_and(
        lhs: GnaIntTensorPrimitive,
        rhs: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        bitwise::bitwise_and(lhs, rhs)
    }

    fn bitwise_and_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
        bitwise::bitwise_and_scalar(lhs, rhs)
    }

    fn bitwise_or(lhs: GnaIntTensorPrimitive, rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        bitwise::bitwise_or(lhs, rhs)
    }

    fn bitwise_or_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
        bitwise::bitwise_or_scalar(lhs, rhs)
    }

    fn bitwise_xor(
        lhs: GnaIntTensorPrimitive,
        rhs: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        bitwise::bitwise_xor(lhs, rhs)
    }

    fn bitwise_xor_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
        bitwise::bitwise_xor_scalar(lhs, rhs)
    }

    fn bitwise_not(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        bitwise::bitwise_not(tensor)
    }

    fn bitwise_left_shift(
        lhs: GnaIntTensorPrimitive,
        rhs: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        bitwise::bitwise_left_shift(lhs, rhs)
    }

    fn bitwise_left_shift_scalar(lhs: GnaIntTensorPrimitive, rhs: Scalar) -> GnaIntTensorPrimitive {
        bitwise::bitwise_left_shift_scalar(lhs, rhs)
    }

    fn bitwise_right_shift(
        lhs: GnaIntTensorPrimitive,
        rhs: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        bitwise::bitwise_right_shift(lhs, rhs)
    }

    fn bitwise_right_shift_scalar(
        lhs: GnaIntTensorPrimitive,
        rhs: Scalar,
    ) -> GnaIntTensorPrimitive {
        bitwise::bitwise_right_shift_scalar(lhs, rhs)
    }

    fn int_sum(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        reduce::int_sum(tensor)
    }

    fn int_sum_dim(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
        reduce::int_sum_dim(tensor, dim)
    }

    fn int_prod(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        reduce::int_prod(tensor)
    }

    fn int_prod_dim(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
        reduce::int_prod_dim(tensor, dim)
    }

    fn int_mean_dim(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
        reduce::int_mean_dim(tensor, dim)
    }

    fn int_cumsum(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
        reduce::int_cumsum(tensor, dim)
    }

    fn int_cumprod(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
        reduce::int_cumprod(tensor, dim)
    }

    fn int_cummin(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
        reduce::int_cummin(tensor, dim)
    }

    fn int_cummax(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
        reduce::int_cummax(tensor, dim)
    }

    fn int_argmax(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
        reduce::int_argmax(tensor, dim)
    }

    fn int_argtopk(tensor: GnaIntTensorPrimitive, dim: usize, k: usize) -> GnaIntTensorPrimitive {
        reduce::int_argtopk(tensor, dim, k)
    }

    fn int_argmin(tensor: GnaIntTensorPrimitive, dim: usize) -> GnaIntTensorPrimitive {
        reduce::int_argmin(tensor, dim)
    }

    fn int_abs(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        unary::int_abs(tensor)
    }

    fn int_unfold(
        tensor: GnaIntTensorPrimitive,
        dim: usize,
        size: usize,
        step: usize,
    ) -> GnaIntTensorPrimitive {
        indexing::int_unfold(tensor, UnfoldParams::new(dim, size, step))
    }

    fn int_equal(
        lhs: GnaIntTensorPrimitive,
        rhs: GnaIntTensorPrimitive,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::int_equal(lhs, rhs, out_dtype)
    }

    fn int_equal_elem(
        lhs: GnaIntTensorPrimitive,
        rhs: Scalar,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::int_equal_elem(lhs, rhs, out_dtype)
    }

    fn int_greater(
        lhs: GnaIntTensorPrimitive,
        rhs: GnaIntTensorPrimitive,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::int_greater(lhs, rhs, out_dtype)
    }

    fn int_greater_elem(
        lhs: GnaIntTensorPrimitive,
        rhs: Scalar,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::int_greater_elem(lhs, rhs, out_dtype)
    }

    fn int_greater_equal(
        lhs: GnaIntTensorPrimitive,
        rhs: GnaIntTensorPrimitive,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::int_greater_equal(lhs, rhs, out_dtype)
    }

    fn int_greater_equal_elem(
        lhs: GnaIntTensorPrimitive,
        rhs: Scalar,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::int_greater_equal_elem(lhs, rhs, out_dtype)
    }

    fn int_lower(
        lhs: GnaIntTensorPrimitive,
        rhs: GnaIntTensorPrimitive,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::int_lower(lhs, rhs, out_dtype)
    }

    fn int_lower_elem(
        lhs: GnaIntTensorPrimitive,
        rhs: Scalar,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::int_lower_elem(lhs, rhs, out_dtype)
    }

    fn int_lower_equal(
        lhs: GnaIntTensorPrimitive,
        rhs: GnaIntTensorPrimitive,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::int_lower_equal(lhs, rhs, out_dtype)
    }

    fn int_lower_equal_elem(
        lhs: GnaIntTensorPrimitive,
        rhs: Scalar,
        out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        comparison::int_lower_equal_elem(lhs, rhs, out_dtype)
    }

    fn int_slice(tensor: GnaIntTensorPrimitive, slices: &[Slice]) -> GnaIntTensorPrimitive {
        indexing::int_slice(tensor, slices)
    }

    fn int_slice_assign(
        tensor: GnaIntTensorPrimitive,
        slices: &[Slice],
        value: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        indexing::int_slice_assign(tensor, slices, value)
    }

    fn int_mask_where(
        tensor: GnaIntTensorPrimitive,
        mask: GnaBoolTensorPrimitive,
        value: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        indexing::int_mask_where(tensor, mask, value)
    }

    fn int_mask_fill(
        tensor: GnaIntTensorPrimitive,
        mask: GnaBoolTensorPrimitive,
        value: Scalar,
    ) -> GnaIntTensorPrimitive {
        indexing::int_mask_fill(tensor, mask, value)
    }

    fn int_gather(
        dim: usize,
        tensor: GnaIntTensorPrimitive,
        indices: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        indexing::int_gather(dim, tensor, indices)
    }

    fn int_scatter(
        dim: usize,
        tensor: GnaIntTensorPrimitive,
        indices: GnaIntTensorPrimitive,
        value: GnaIntTensorPrimitive,
        update: IndexingUpdateOp,
    ) -> GnaIntTensorPrimitive {
        indexing::int_scatter(dim, tensor, indices, value, update)
    }

    fn int_select(
        tensor: GnaIntTensorPrimitive,
        dim: usize,
        indices: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        indexing::int_select(tensor, dim, indices)
    }

    fn int_select_assign(
        tensor: GnaIntTensorPrimitive,
        dim: usize,
        indices: GnaIntTensorPrimitive,
        value: GnaIntTensorPrimitive,
        update: IndexingUpdateOp,
    ) -> GnaIntTensorPrimitive {
        indexing::int_select_assign(tensor, dim, indices, value, update)
    }

    fn int_swap_dims(
        tensor: GnaIntTensorPrimitive,
        dim1: usize,
        dim2: usize,
    ) -> GnaIntTensorPrimitive {
        indexing::int_swap_dims(tensor, dim1, dim2)
    }

    fn int_permute(tensor: GnaIntTensorPrimitive, axes: &[usize]) -> GnaIntTensorPrimitive {
        indexing::int_permute(tensor, axes)
    }

    fn int_flip(tensor: GnaIntTensorPrimitive, axes: &[usize]) -> GnaIntTensorPrimitive {
        indexing::int_flip(tensor, axes)
    }
}
