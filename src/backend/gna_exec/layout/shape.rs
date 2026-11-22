use burn_backend::Shape;

/// Borrowed view of a raw byte tensor with its shape and element size.
#[derive(Clone, Copy)]
pub struct ByteTensorView<'a> {
    bytes: &'a [u8],
    dims: &'a [usize],
    elem_size: usize,
}

impl<'a> ByteTensorView<'a> {
    pub const fn new(bytes: &'a [u8], dims: &'a [usize], elem_size: usize) -> Self {
        Self {
            bytes,
            dims,
            elem_size,
        }
    }

    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub const fn dims(&self) -> &'a [usize] {
        self.dims
    }

    pub const fn elem_size(&self) -> usize {
        self.elem_size
    }
}

pub(super) fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

pub(super) fn linear_to_coords(mut idx: usize, dims: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; dims.len()];
    for i in (0..dims.len()).rev() {
        let d = dims[i].max(1);
        coords[i] = idx % d;
        idx /= d;
    }
    coords
}

pub(super) fn coords_to_linear(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides.iter())
        .map(|(&c, &s)| c * s)
        .sum()
}

pub fn permute_bytes(src: ByteTensorView<'_>, axes: &[usize]) -> (Shape, Vec<u8>) {
    let ByteTensorView {
        bytes,
        dims,
        elem_size,
    } = src;
    let out_dims: Vec<usize> = axes.iter().map(|&a| dims[a]).collect();
    let total: usize = out_dims.iter().product();
    let in_strides = compute_strides(dims);
    let mut out = vec![0u8; total * elem_size];

    for out_idx in 0..total {
        let out_coords = linear_to_coords(out_idx, &out_dims);
        let mut in_coords = vec![0; dims.len()];
        for (o_dim, &axis) in axes.iter().enumerate() {
            in_coords[axis] = out_coords[o_dim];
        }
        let in_idx = coords_to_linear(&in_coords, &in_strides);
        out[out_idx * elem_size..(out_idx + 1) * elem_size]
            .copy_from_slice(&bytes[in_idx * elem_size..(in_idx + 1) * elem_size]);
    }

    (Shape::from(out_dims), out)
}

pub fn swap_dims_bytes(src: ByteTensorView<'_>, dim1: usize, dim2: usize) -> (Shape, Vec<u8>) {
    let mut axes: Vec<usize> = (0..src.dims.len()).collect();
    axes.swap(dim1, dim2);
    permute_bytes(src, &axes)
}

pub fn flip_bytes(src: ByteTensorView<'_>, axes: &[usize]) -> Vec<u8> {
    let ByteTensorView {
        bytes,
        dims,
        elem_size,
    } = src;
    let total: usize = dims.iter().product();
    let strides = compute_strides(dims);
    let mut out = vec![0u8; total * elem_size];

    for out_idx in 0..total {
        let mut coords = linear_to_coords(out_idx, dims);
        for &axis in axes {
            if axis < dims.len() {
                coords[axis] = dims[axis] - 1 - coords[axis];
            }
        }
        let in_idx = coords_to_linear(&coords, &strides);
        out[out_idx * elem_size..(out_idx + 1) * elem_size]
            .copy_from_slice(&bytes[in_idx * elem_size..(in_idx + 1) * elem_size]);
    }
    out
}

pub fn expand_bytes(src: ByteTensorView<'_>, out_shape: &Shape) -> Vec<u8> {
    let ByteTensorView {
        bytes,
        dims: in_dims,
        elem_size,
    } = src;
    let out_dims: &[usize] = out_shape;
    let total: usize = out_dims.iter().product();
    let in_strides = compute_strides(in_dims);
    let rank_diff = out_dims.len().saturating_sub(in_dims.len());
    let mut out = vec![0u8; total * elem_size];

    for out_idx in 0..total {
        let out_coords = linear_to_coords(out_idx, out_dims);
        let mut in_coords = vec![0; in_dims.len()];
        for (i, &in_d) in in_dims.iter().enumerate() {
            in_coords[i] = if in_d == 1 {
                0
            } else {
                out_coords[i + rank_diff]
            };
        }
        let in_idx = coords_to_linear(&in_coords, &in_strides);
        out[out_idx * elem_size..(out_idx + 1) * elem_size]
            .copy_from_slice(&bytes[in_idx * elem_size..(in_idx + 1) * elem_size]);
    }
    out
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Shape {
    let max_rank = a.len().max(b.len());
    let mut out = vec![1usize; max_rank];
    for i in 0..max_rank {
        let da = if i < max_rank - a.len() {
            1
        } else {
            a[i - (max_rank - a.len())]
        };
        let db = if i < max_rank - b.len() {
            1
        } else {
            b[i - (max_rank - b.len())]
        };
        out[i] = da.max(db);
    }
    Shape::from(out)
}

pub fn broadcast_pair_bytes(
    a: ByteTensorView<'_>,
    b: ByteTensorView<'_>,
) -> (Shape, Vec<u8>, Vec<u8>) {
    let out_shape = broadcast_shape(a.dims, b.dims);
    let a_out = if a.dims == out_shape.as_slice() {
        a.bytes.to_vec()
    } else {
        expand_bytes(a, &out_shape)
    };
    let b_out = if b.dims == out_shape.as_slice() {
        b.bytes.to_vec()
    } else {
        expand_bytes(b, &out_shape)
    };
    (out_shape, a_out, b_out)
}
