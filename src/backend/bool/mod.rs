use super::GnaBackend;
use crate::device::GnaDevice;
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::ops::BoolTensorOps;
use burn_backend::{
    BoolStore, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
};

mod creation;
mod indexing;
mod logical;

impl BoolTensorOps<Self> for GnaBackend {
    fn bool_empty(shape: Shape, device: &GnaDevice, dtype: BoolStore) -> GnaBoolTensorPrimitive {
        creation::bool_empty(shape, device, dtype)
    }

    fn bool_zeros(shape: Shape, device: &GnaDevice, dtype: BoolStore) -> GnaBoolTensorPrimitive {
        creation::bool_zeros(shape, device, dtype)
    }

    fn bool_ones(shape: Shape, device: &GnaDevice, dtype: BoolStore) -> GnaBoolTensorPrimitive {
        creation::bool_ones(shape, device, dtype)
    }

    async fn bool_into_data(tensor: GnaBoolTensorPrimitive) -> Result<TensorData, ExecutionError> {
        creation::bool_into_data(tensor).await
    }

    fn bool_from_data(data: TensorData, device: &GnaDevice) -> GnaBoolTensorPrimitive {
        creation::bool_from_data(data, device)
    }

    fn bool_into_int(tensor: GnaBoolTensorPrimitive, out_dtype: IntDType) -> GnaIntTensorPrimitive {
        creation::bool_into_int(tensor, out_dtype)
    }

    fn bool_into_float(
        tensor: GnaBoolTensorPrimitive,
        out_dtype: FloatDType,
    ) -> GnaFloatTensorPrimitive {
        creation::bool_into_float(tensor, out_dtype)
    }

    fn bool_to_device(
        tensor: GnaBoolTensorPrimitive,
        device: &GnaDevice,
    ) -> GnaBoolTensorPrimitive {
        creation::bool_to_device(tensor, device)
    }

    fn bool_reshape(tensor: GnaBoolTensorPrimitive, shape: Shape) -> GnaBoolTensorPrimitive {
        creation::bool_reshape(tensor, shape)
    }

    fn bool_slice(tensor: GnaBoolTensorPrimitive, slices: &[Slice]) -> GnaBoolTensorPrimitive {
        indexing::bool_slice(tensor, slices)
    }

    fn bool_slice_assign(
        tensor: GnaBoolTensorPrimitive,
        slices: &[Slice],
        value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        indexing::bool_slice_assign(tensor, slices, value)
    }

    fn bool_mask_where(
        tensor: GnaBoolTensorPrimitive,
        mask: GnaBoolTensorPrimitive,
        value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        indexing::bool_mask_where(tensor, mask, value)
    }

    fn bool_mask_fill(
        tensor: GnaBoolTensorPrimitive,
        mask: GnaBoolTensorPrimitive,
        value: Scalar,
    ) -> GnaBoolTensorPrimitive {
        indexing::bool_mask_fill(tensor, mask, value)
    }

    fn bool_gather(
        dim: usize,
        tensor: GnaBoolTensorPrimitive,
        indices: GnaIntTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        indexing::bool_gather(dim, tensor, indices)
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: GnaBoolTensorPrimitive,
        indices: GnaIntTensorPrimitive,
        value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        indexing::bool_scatter_or(dim, tensor, indices, value)
    }

    fn bool_select(
        tensor: GnaBoolTensorPrimitive,
        dim: usize,
        indices: GnaIntTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        indexing::bool_select(tensor, dim, indices)
    }

    fn bool_select_or(
        tensor: GnaBoolTensorPrimitive,
        dim: usize,
        indices: GnaIntTensorPrimitive,
        value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        indexing::bool_select_or(tensor, dim, indices, value)
    }

    fn bool_equal(
        lhs: GnaBoolTensorPrimitive,
        rhs: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        logical::bool_equal(lhs, rhs)
    }

    fn bool_equal_elem(lhs: GnaBoolTensorPrimitive, rhs: Scalar) -> GnaBoolTensorPrimitive {
        logical::bool_equal_elem(lhs, rhs)
    }

    fn bool_not(tensor: GnaBoolTensorPrimitive) -> GnaBoolTensorPrimitive {
        logical::bool_not(tensor)
    }

    fn bool_and(
        lhs: GnaBoolTensorPrimitive,
        rhs: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        logical::bool_and(lhs, rhs)
    }

    fn bool_or(lhs: GnaBoolTensorPrimitive, rhs: GnaBoolTensorPrimitive) -> GnaBoolTensorPrimitive {
        logical::bool_or(lhs, rhs)
    }

    fn bool_swap_dims(
        tensor: GnaBoolTensorPrimitive,
        dim1: usize,
        dim2: usize,
    ) -> GnaBoolTensorPrimitive {
        indexing::bool_swap_dims(tensor, dim1, dim2)
    }

    fn bool_permute(tensor: GnaBoolTensorPrimitive, axes: &[usize]) -> GnaBoolTensorPrimitive {
        indexing::bool_permute(tensor, axes)
    }

    fn bool_flip(tensor: GnaBoolTensorPrimitive, axes: &[usize]) -> GnaBoolTensorPrimitive {
        indexing::bool_flip(tensor, axes)
    }

    fn bool_expand(tensor: GnaBoolTensorPrimitive, shape: Shape) -> GnaBoolTensorPrimitive {
        creation::bool_expand(tensor, shape)
    }

    fn bool_unfold(
        tensor: GnaBoolTensorPrimitive,
        dim: usize,
        size: usize,
        step: usize,
    ) -> GnaBoolTensorPrimitive {
        indexing::bool_unfold(tensor, dim, size, step)
    }
}
