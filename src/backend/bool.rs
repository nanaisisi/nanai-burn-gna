use super::GnaBackend;
use crate::device::GnaDevice;
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::ops::BoolTensorOps;
use burn_backend::{
    BoolStore, DType, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
};

// ---------------------------------------------------------------------------
// BoolTensorOps
// ---------------------------------------------------------------------------
impl BoolTensorOps<Self> for GnaBackend {
    fn bool_empty(shape: Shape, _device: &GnaDevice, _dtype: BoolStore) -> GnaBoolTensorPrimitive {
        let num = shape.num_elements();
        GnaBoolTensorPrimitive::new(shape, GnaBuffer::zeros(num))
    }

    fn bool_zeros(shape: Shape, device: &GnaDevice, dtype: BoolStore) -> GnaBoolTensorPrimitive {
        Self::bool_empty(shape, device, dtype)
    }

    fn bool_ones(shape: Shape, _device: &GnaDevice, _dtype: BoolStore) -> GnaBoolTensorPrimitive {
        let num = shape.num_elements();
        GnaBoolTensorPrimitive::new(shape, GnaBuffer::filled(1, num))
    }

    async fn bool_into_data(tensor: GnaBoolTensorPrimitive) -> Result<TensorData, ExecutionError> {
        let (shape, buffer) = tensor.into_parts();
        let data =
            TensorData::from_bytes_vec(buffer.into_vec(), shape, DType::Bool(BoolStore::Native));
        Ok(data)
    }

    fn bool_from_data(data: TensorData, _device: &GnaDevice) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive::new(
            data.shape().clone(),
            GnaBuffer::from_bytes(data.into_bytes().to_vec()),
        )
    }

    fn bool_into_int(tensor: GnaBoolTensorPrimitive, out_dtype: IntDType) -> GnaIntTensorPrimitive {
        let (shape, buffer) = tensor.into_parts();
        let vals: Vec<i64> = buffer
            .as_slice()
            .iter()
            .map(|&b| i64::from(b != 0))
            .collect();
        GnaIntTensorPrimitive::new(
            shape,
            out_dtype,
            GnaBuffer::from_bytes(super::gna_exec::encode_ints(&vals, out_dtype)),
        )
    }

    fn bool_into_float(
        tensor: GnaBoolTensorPrimitive,
        out_dtype: FloatDType,
    ) -> GnaFloatTensorPrimitive {
        let (shape, buffer) = tensor.into_parts();
        let vals: Vec<f32> = buffer
            .as_slice()
            .iter()
            .map(|&b| if b != 0 { 1.0 } else { 0.0 })
            .collect();
        GnaFloatTensorPrimitive::new(
            shape,
            out_dtype,
            GnaBuffer::from_bytes(super::gna_exec::encode_floats(&vals, out_dtype)),
        )
    }

    fn bool_to_device(
        tensor: GnaBoolTensorPrimitive,
        _device: &GnaDevice,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_reshape(tensor: GnaBoolTensorPrimitive, shape: Shape) -> GnaBoolTensorPrimitive {
        tensor.with_shape(shape)
    }

    fn bool_slice(tensor: GnaBoolTensorPrimitive, slices: &[Slice]) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let (shape, bytes) = super::gna_exec::layout::slice_bytes(
            tensor.buffer().as_slice(),
            &tensor.shape(),
            slices,
            1,
        );
        GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
    }

    fn bool_slice_assign(
        tensor: GnaBoolTensorPrimitive,
        slices: &[Slice],
        value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let bytes = super::gna_exec::layout::slice_assign_bytes(
            tensor.buffer().as_slice(),
            &tensor.shape(),
            slices,
            value.buffer().as_slice(),
            1,
        );
        GnaBoolTensorPrimitive::new(tensor.shape(), GnaBuffer::from_bytes(bytes))
    }

    fn bool_mask_where(
        tensor: GnaBoolTensorPrimitive,
        mask: GnaBoolTensorPrimitive,
        value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let bytes = super::gna_exec::layout::mask_where_bytes(
            tensor.buffer().as_slice(),
            mask.buffer().as_slice(),
            value.buffer().as_slice(),
            1,
        );
        GnaBoolTensorPrimitive::new(tensor.shape(), GnaBuffer::from_bytes(bytes))
    }

    fn bool_mask_fill(
        tensor: GnaBoolTensorPrimitive,
        mask: GnaBoolTensorPrimitive,
        value: Scalar,
    ) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let fill = [u8::from(value.elem::<bool>())];
        let bytes = super::gna_exec::layout::mask_fill_bytes(
            tensor.buffer().as_slice(),
            mask.buffer().as_slice(),
            &fill,
            1,
        );
        GnaBoolTensorPrimitive::new(tensor.shape(), GnaBuffer::from_bytes(bytes))
    }

    fn bool_gather(
        dim: usize,
        tensor: GnaBoolTensorPrimitive,
        indices: GnaIntTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let idx_vals = super::gna_exec::decode_ints(indices.buffer().as_slice(), indices.dtype());
        let (shape, bytes) = super::gna_exec::layout::gather_bytes(
            dim,
            tensor.buffer().as_slice(),
            &tensor.shape(),
            &idx_vals,
            &indices.shape(),
            1,
        );
        GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: GnaBoolTensorPrimitive,
        indices: GnaIntTensorPrimitive,
        value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        super::gna_exec::layout::scatter_or_bool(dim, tensor, indices, value)
    }

    fn bool_select(
        tensor: GnaBoolTensorPrimitive,
        dim: usize,
        indices: GnaIntTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let idx_vals = super::gna_exec::decode_ints(indices.buffer().as_slice(), indices.dtype());
        let (shape, bytes) = super::gna_exec::layout::select_bytes(
            tensor.buffer().as_slice(),
            &tensor.shape(),
            dim,
            &idx_vals,
            1,
        );
        GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
    }

    fn bool_select_or(
        tensor: GnaBoolTensorPrimitive,
        dim: usize,
        indices: GnaIntTensorPrimitive,
        value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        super::gna_exec::layout::select_or_bool(tensor, dim, indices, value)
    }

    fn bool_equal(
        lhs: GnaBoolTensorPrimitive,
        rhs: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let out: Vec<u8> = lhs
            .buffer()
            .as_slice()
            .iter()
            .zip(rhs.buffer().as_slice().iter())
            .map(|(&a, &b)| u8::from((a != 0) == (b != 0)))
            .collect();
        GnaBoolTensorPrimitive::new(lhs.shape(), GnaBuffer::from_bytes(out))
    }

    fn bool_equal_elem(lhs: GnaBoolTensorPrimitive, rhs: Scalar) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let r = rhs.elem::<bool>();
        let out: Vec<u8> = lhs
            .buffer()
            .as_slice()
            .iter()
            .map(|&a| u8::from((a != 0) == r))
            .collect();
        GnaBoolTensorPrimitive::new(lhs.shape(), GnaBuffer::from_bytes(out))
    }

    fn bool_not(tensor: GnaBoolTensorPrimitive) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let out: Vec<u8> = tensor
            .buffer()
            .as_slice()
            .iter()
            .map(|&a| u8::from(a == 0))
            .collect();
        GnaBoolTensorPrimitive::new(tensor.shape(), GnaBuffer::from_bytes(out))
    }

    fn bool_and(
        lhs: GnaBoolTensorPrimitive,
        rhs: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let out: Vec<u8> = lhs
            .buffer()
            .as_slice()
            .iter()
            .zip(rhs.buffer().as_slice().iter())
            .map(|(&a, &b)| u8::from(a != 0 && b != 0))
            .collect();
        GnaBoolTensorPrimitive::new(lhs.shape(), GnaBuffer::from_bytes(out))
    }

    fn bool_or(lhs: GnaBoolTensorPrimitive, rhs: GnaBoolTensorPrimitive) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let out: Vec<u8> = lhs
            .buffer()
            .as_slice()
            .iter()
            .zip(rhs.buffer().as_slice().iter())
            .map(|(&a, &b)| u8::from(a != 0 || b != 0))
            .collect();
        GnaBoolTensorPrimitive::new(lhs.shape(), GnaBuffer::from_bytes(out))
    }

    fn bool_swap_dims(
        tensor: GnaBoolTensorPrimitive,
        dim1: usize,
        dim2: usize,
    ) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let (shape, bytes) = super::gna_exec::layout::swap_dims_bytes(
            tensor.buffer().as_slice(),
            &tensor.shape(),
            dim1,
            dim2,
            1,
        );
        GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
    }

    fn bool_permute(tensor: GnaBoolTensorPrimitive, axes: &[usize]) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let (shape, bytes) = super::gna_exec::layout::permute_bytes(
            tensor.buffer().as_slice(),
            &tensor.shape(),
            axes,
            1,
        );
        GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
    }

    fn bool_flip(tensor: GnaBoolTensorPrimitive, axes: &[usize]) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let bytes = super::gna_exec::layout::flip_bytes(
            tensor.buffer().as_slice(),
            &tensor.shape(),
            axes,
            1,
        );
        GnaBoolTensorPrimitive::new(tensor.shape(), GnaBuffer::from_bytes(bytes))
    }

    fn bool_expand(tensor: GnaBoolTensorPrimitive, shape: Shape) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let bytes = super::gna_exec::layout::expand_bytes(
            tensor.buffer().as_slice(),
            &tensor.shape(),
            &shape,
            1,
        );
        GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
    }

    fn bool_unfold(
        tensor: GnaBoolTensorPrimitive,
        dim: usize,
        size: usize,
        step: usize,
    ) -> GnaBoolTensorPrimitive {
        use burn_backend::TensorMetadata;
        let (shape, bytes) = super::gna_exec::layout::unfold_bytes(
            tensor.buffer().as_slice(),
            &tensor.shape(),
            super::UnfoldParams::new(dim, size, step),
            1,
        );
        GnaBoolTensorPrimitive::new(shape, GnaBuffer::from_bytes(bytes))
    }
}
