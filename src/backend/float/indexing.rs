use crate::backend::UnfoldParams;
use crate::backend::gna_exec::decode_ints;
use crate::backend::gna_exec::layout;
use crate::tensor::bool::GnaBoolTensorPrimitive;
use crate::tensor::buffer::GnaBuffer;
use crate::tensor::float::GnaFloatTensorPrimitive;
use crate::tensor::int::GnaIntTensorPrimitive;
use burn_backend::tensor::IndexingUpdateOp;
use burn_backend::{DType, Scalar, Slice, TensorMetadata};

pub(super) fn float_swap_dims(
    tensor: GnaFloatTensorPrimitive,
    dim1: usize,
    dim2: usize,
) -> GnaFloatTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let (out_shape, out_bytes) = layout::swap_dims_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        dim1,
        dim2,
        elem_size,
    );
    GnaFloatTensorPrimitive::new(out_shape, tensor.dtype(), GnaBuffer::from_bytes(out_bytes))
}

pub(super) fn float_permute(
    tensor: GnaFloatTensorPrimitive,
    axes: &[usize],
) -> GnaFloatTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let (out_shape, out_bytes) =
        layout::permute_bytes(tensor.buffer().as_slice(), &tensor.shape(), axes, elem_size);
    GnaFloatTensorPrimitive::new(out_shape, tensor.dtype(), GnaBuffer::from_bytes(out_bytes))
}

pub(super) fn float_flip(
    tensor: GnaFloatTensorPrimitive,
    axes: &[usize],
) -> GnaFloatTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let out_bytes =
        layout::flip_bytes(tensor.buffer().as_slice(), &tensor.shape(), axes, elem_size);
    GnaFloatTensorPrimitive::new(
        tensor.shape(),
        tensor.dtype(),
        GnaBuffer::from_bytes(out_bytes),
    )
}

pub(super) fn float_gather(
    dim: usize,
    tensor: GnaFloatTensorPrimitive,
    indices: GnaIntTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let idx_vals = decode_ints(indices.buffer().as_slice(), indices.dtype());
    let (out_shape, out_bytes) = layout::gather_bytes(
        dim,
        tensor.buffer().as_slice(),
        &tensor.shape(),
        &idx_vals,
        &indices.shape(),
        elem_size,
    );
    GnaFloatTensorPrimitive::new(out_shape, tensor.dtype(), GnaBuffer::from_bytes(out_bytes))
}

pub(super) fn float_scatter(
    dim: usize,
    tensor: GnaFloatTensorPrimitive,
    indices: GnaIntTensorPrimitive,
    value: GnaFloatTensorPrimitive,
    update: IndexingUpdateOp,
) -> GnaFloatTensorPrimitive {
    layout::scatter_op_f32(dim, tensor, indices, value, update)
}

pub(super) fn float_select(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
    indices: GnaIntTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let idx_vals = decode_ints(indices.buffer().as_slice(), indices.dtype());
    let (out_shape, out_bytes) = layout::select_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        dim,
        &idx_vals,
        elem_size,
    );
    GnaFloatTensorPrimitive::new(out_shape, tensor.dtype(), GnaBuffer::from_bytes(out_bytes))
}

pub(super) fn float_select_assign(
    tensor: GnaFloatTensorPrimitive,
    dim: usize,
    indices: GnaIntTensorPrimitive,
    value: GnaFloatTensorPrimitive,
    update: IndexingUpdateOp,
) -> GnaFloatTensorPrimitive {
    layout::select_assign_op_f32(tensor, dim, indices, value, update)
}

pub(super) fn float_slice(
    tensor: GnaFloatTensorPrimitive,
    slices: &[Slice],
) -> GnaFloatTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let (out_shape, out_bytes) = layout::slice_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        slices,
        elem_size,
    );
    GnaFloatTensorPrimitive::new(out_shape, tensor.dtype(), GnaBuffer::from_bytes(out_bytes))
}

pub(super) fn float_slice_assign(
    tensor: GnaFloatTensorPrimitive,
    slices: &[Slice],
    value: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let out_bytes = layout::slice_assign_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        slices,
        value.buffer().as_slice(),
        elem_size,
    );
    GnaFloatTensorPrimitive::new(
        tensor.shape(),
        tensor.dtype(),
        GnaBuffer::from_bytes(out_bytes),
    )
}

pub(super) fn float_mask_where(
    tensor: GnaFloatTensorPrimitive,
    mask: GnaBoolTensorPrimitive,
    value: GnaFloatTensorPrimitive,
) -> GnaFloatTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let out_bytes = layout::mask_where_bytes(
        tensor.buffer().as_slice(),
        mask.buffer().as_slice(),
        value.buffer().as_slice(),
        elem_size,
    );
    GnaFloatTensorPrimitive::new(
        tensor.shape(),
        tensor.dtype(),
        GnaBuffer::from_bytes(out_bytes),
    )
}

pub(super) fn float_mask_fill(
    tensor: GnaFloatTensorPrimitive,
    mask: GnaBoolTensorPrimitive,
    value: Scalar,
) -> GnaFloatTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let fill = layout::scalar_to_le_bytes(value, elem_size);
    let out_bytes = layout::mask_fill_bytes(
        tensor.buffer().as_slice(),
        mask.buffer().as_slice(),
        &fill,
        elem_size,
    );
    GnaFloatTensorPrimitive::new(
        tensor.shape(),
        tensor.dtype(),
        GnaBuffer::from_bytes(out_bytes),
    )
}

pub(super) fn float_unfold(
    tensor: GnaFloatTensorPrimitive,
    params: UnfoldParams,
) -> GnaFloatTensorPrimitive {
    let elem_size = DType::from(tensor.dtype()).size();
    let (out_shape, out_bytes) = layout::unfold_bytes(
        tensor.buffer().as_slice(),
        &tensor.shape(),
        params,
        elem_size,
    );
    GnaFloatTensorPrimitive::new(out_shape, tensor.dtype(), GnaBuffer::from_bytes(out_bytes))
}
