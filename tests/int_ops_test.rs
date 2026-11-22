use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use burn_backend::ops::{BoolTensorOps, FloatTensorOps, IntTensorOps};
use burn_backend::tensor::IndexingUpdateOp;
use burn_backend::{
    BoolStore, DType, Distribution, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
    TensorMetadata,
};
use nanai_burn_gna::{GnaBackend, GnaBuffer, GnaDevice, GnaIntTensorPrimitive};

fn block_on<F: Future>(mut fut: F) -> F::Output {
    fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    fn noop(_: *const ()) {}
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    let mut fut = unsafe { Pin::new_unchecked(&mut fut) };
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn dev() -> GnaDevice {
    GnaDevice::default()
}

fn int_tensor(data: Vec<i32>, shape: [usize; 2]) -> GnaIntTensorPrimitive {
    GnaBackend::int_from_data(TensorData::new(data, shape), &dev())
}

fn int_tensor_1d(data: Vec<i32>) -> GnaIntTensorPrimitive {
    let len = data.len();
    GnaBackend::int_from_data(TensorData::new(data, [len]), &dev())
}

fn s(v: i32) -> Scalar {
    Scalar::new(v, &DType::I32)
}

fn read_i32(t: GnaIntTensorPrimitive) -> Vec<i32> {
    block_on(GnaBackend::int_into_data(t))
        .unwrap()
        .try_to_vec::<i32>()
        .unwrap()
}

fn read_bool(t: nanai_burn_gna::GnaBoolTensorPrimitive) -> Vec<bool> {
    block_on(GnaBackend::bool_into_data(t))
        .unwrap()
        .try_to_vec::<bool>()
        .unwrap()
}

#[test]
fn test_gna_int_tensor_primitive_methods() {
    let buf = GnaBuffer::zeros(16);
    let prim = GnaIntTensorPrimitive::new(Shape::new([2, 2]), IntDType::I32, buf);
    assert_eq!(prim.dtype(), IntDType::I32);
    assert_eq!(TensorMetadata::dtype(&prim), DType::I32);
    assert_eq!(prim.shape(), Shape::new([2, 2]));
    assert_eq!(prim.device(), GnaDevice::default());
    assert!(prim.can_mut());
    assert_eq!(prim.buffer().as_slice().len(), 16);

    let reshaped = prim.clone().with_shape(Shape::new([4]));
    assert_eq!(reshaped.shape(), Shape::new([4]));

    let recasted = prim.clone().with_dtype(IntDType::I16);
    assert_eq!(recasted.dtype(), IntDType::I16);

    let (shape, dtype, buf2) = prim.clone().into_parts();
    assert_eq!(shape, Shape::new([2, 2]));
    assert_eq!(dtype, IntDType::I32);
    assert_eq!(buf2.as_slice().len(), 16);

    let buf3 = prim.into_buffer();
    assert_eq!(buf3.as_slice().len(), 16);
}

#[test]
fn test_int_creation_and_conversion() {
    let empty = GnaBackend::int_empty(Shape::new([2, 3]), &dev(), IntDType::I32);
    assert_eq!(empty.shape(), Shape::new([2, 3]));
    assert_eq!(read_i32(empty), vec![0; 6]);

    let t = int_tensor(vec![1, 2, 3, 4], [2, 2]);
    let t = GnaBackend::int_to_device(t, &dev());
    let reshaped = GnaBackend::int_reshape(t.clone(), Shape::new([4]));
    assert_eq!(reshaped.shape(), Shape::new([4]));
    assert_eq!(read_i32(reshaped), vec![1, 2, 3, 4]);

    let casted = GnaBackend::int_cast(t.clone(), IntDType::I16);
    assert_eq!(casted.dtype(), IntDType::I16);
    let back = GnaBackend::int_cast(casted, IntDType::I32);
    assert_eq!(read_i32(back), vec![1, 2, 3, 4]);
    // Same-dtype cast fast path
    let same = GnaBackend::int_cast(t.clone(), IntDType::I32);
    assert_eq!(read_i32(same), vec![1, 2, 3, 4]);

    let flt = GnaBackend::int_into_float(t, FloatDType::F32);
    let flt_data = block_on(GnaBackend::float_into_data(flt))
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(flt_data, vec![1.0, 2.0, 3.0, 4.0]);

    let single = int_tensor(vec![7, 8], [1, 2]);
    let expanded = GnaBackend::int_expand(single, Shape::new([3, 2]));
    assert_eq!(expanded.shape(), Shape::new([3, 2]));
    assert_eq!(read_i32(expanded), vec![7, 8, 7, 8, 7, 8]);

    let rand_t = GnaBackend::int_random(
        Shape::new([4]),
        Distribution::Uniform(0.0, 10.0),
        &dev(),
        IntDType::I32,
    );
    assert_eq!(rand_t.shape(), Shape::new([4]));
    assert_eq!(read_i32(rand_t).len(), 4);
}

#[test]
fn test_int_arithmetic_and_matmul() {
    let a = int_tensor_1d(vec![10, 20, -30, 40]);
    let b = int_tensor_1d(vec![1, 2, 3, 4]);

    assert_eq!(
        read_i32(GnaBackend::int_add(a.clone(), b.clone())),
        vec![11, 22, -27, 44]
    );
    assert_eq!(
        read_i32(GnaBackend::int_add_scalar(a.clone(), s(5))),
        vec![15, 25, -25, 45]
    );
    assert_eq!(
        read_i32(GnaBackend::int_sub(a.clone(), b.clone())),
        vec![9, 18, -33, 36]
    );
    assert_eq!(
        read_i32(GnaBackend::int_sub_scalar(a.clone(), s(5))),
        vec![5, 15, -35, 35]
    );
    assert_eq!(
        read_i32(GnaBackend::int_mul(a.clone(), b.clone())),
        vec![10, 40, -90, 160]
    );
    assert_eq!(
        read_i32(GnaBackend::int_mul_scalar(a.clone(), s(2))),
        vec![20, 40, -60, 80]
    );
    assert_eq!(
        read_i32(GnaBackend::int_div(a.clone(), b.clone())),
        vec![10, 10, -10, 10]
    );
    assert_eq!(
        read_i32(GnaBackend::int_div_scalar(a.clone(), s(10))),
        vec![1, 2, -3, 4]
    );
    assert_eq!(
        read_i32(GnaBackend::int_remainder(a.clone(), b)),
        vec![0, 0, 0, 0]
    );
    assert_eq!(
        read_i32(GnaBackend::int_remainder_scalar(a, s(7))),
        vec![3, 6, -2, 5]
    );

    let m1 = int_tensor(vec![1, 2, 3, 4], [2, 2]);
    let m2 = int_tensor(vec![5, 6, 7, 8], [2, 2]);
    let mm = GnaBackend::int_matmul(m1, m2);
    assert_eq!(mm.shape(), Shape::new([2, 2]));
    assert_eq!(read_i32(mm), vec![19, 22, 43, 50]);
}

#[test]
fn test_int_bitwise_and_unary() {
    let a = int_tensor_1d(vec![0b1010, 0b1100, -5, 8]);
    let b = int_tensor_1d(vec![0b1100, 0b1010, 3, 1]);

    assert_eq!(
        read_i32(GnaBackend::bitwise_and(a.clone(), b.clone())),
        vec![0b1000, 0b1000, -5 & 3, 0]
    );
    assert_eq!(
        read_i32(GnaBackend::bitwise_and_scalar(a.clone(), s(0b1111))),
        vec![0b1010, 0b1100, (-5i32) & 0b1111, 8]
    );
    assert_eq!(
        read_i32(GnaBackend::bitwise_or(a.clone(), b.clone())),
        vec![0b1110, 0b1110, -5 | 3, 9]
    );
    assert_eq!(
        read_i32(GnaBackend::bitwise_or_scalar(a.clone(), s(1))),
        vec![0b1011, 0b1101, -5 | 1, 9]
    );
    assert_eq!(
        read_i32(GnaBackend::bitwise_xor(a.clone(), b.clone())),
        vec![0b0110, 0b0110, -5 ^ 3, 9]
    );
    assert_eq!(
        read_i32(GnaBackend::bitwise_xor_scalar(a.clone(), s(0b1111))),
        vec![0b0101, 0b0011, -5 ^ 0b1111, 7]
    );
    assert_eq!(
        read_i32(GnaBackend::bitwise_not(a.clone())),
        vec![!0b1010, !0b1100, !(-5), !8]
    );

    let sh_lhs = int_tensor_1d(vec![1, 2, 16, 32]);
    let sh_rhs = int_tensor_1d(vec![1, 2, 1, 2]);
    assert_eq!(
        read_i32(GnaBackend::bitwise_left_shift(
            sh_lhs.clone(),
            sh_rhs.clone()
        )),
        vec![2, 8, 32, 128]
    );
    assert_eq!(
        read_i32(GnaBackend::bitwise_left_shift_scalar(sh_lhs.clone(), s(1))),
        vec![2, 4, 32, 64]
    );
    assert_eq!(
        read_i32(GnaBackend::bitwise_right_shift(sh_lhs.clone(), sh_rhs)),
        vec![0, 0, 8, 8]
    );
    assert_eq!(
        read_i32(GnaBackend::bitwise_right_shift_scalar(sh_lhs, s(1))),
        vec![0, 1, 8, 16]
    );

    assert_eq!(
        read_i32(GnaBackend::int_abs(int_tensor_1d(vec![-10, 0, 15, -1]))),
        vec![10, 0, 15, 1]
    );
}

#[test]
fn test_int_comparison() {
    let a = int_tensor_1d(vec![1, 5, 10]);
    let b = int_tensor_1d(vec![2, 5, 8]);

    assert_eq!(
        read_bool(GnaBackend::int_equal(
            a.clone(),
            b.clone(),
            BoolStore::Native
        )),
        vec![false, true, false]
    );
    assert_eq!(
        read_bool(GnaBackend::int_equal_elem(
            a.clone(),
            s(5),
            BoolStore::Native
        )),
        vec![false, true, false]
    );
    assert_eq!(
        read_bool(GnaBackend::int_greater(
            a.clone(),
            b.clone(),
            BoolStore::Native
        )),
        vec![false, false, true]
    );
    assert_eq!(
        read_bool(GnaBackend::int_greater_elem(
            a.clone(),
            s(5),
            BoolStore::Native
        )),
        vec![false, false, true]
    );
    assert_eq!(
        read_bool(GnaBackend::int_greater_equal(
            a.clone(),
            b.clone(),
            BoolStore::Native
        )),
        vec![false, true, true]
    );
    assert_eq!(
        read_bool(GnaBackend::int_greater_equal_elem(
            a.clone(),
            s(5),
            BoolStore::Native
        )),
        vec![false, true, true]
    );
    assert_eq!(
        read_bool(GnaBackend::int_lower(
            a.clone(),
            b.clone(),
            BoolStore::Native
        )),
        vec![true, false, false]
    );
    assert_eq!(
        read_bool(GnaBackend::int_lower_elem(
            a.clone(),
            s(5),
            BoolStore::Native
        )),
        vec![true, false, false]
    );
    assert_eq!(
        read_bool(GnaBackend::int_lower_equal(a.clone(), b, BoolStore::Native)),
        vec![true, true, false]
    );
    assert_eq!(
        read_bool(GnaBackend::int_lower_equal_elem(a, s(5), BoolStore::Native)),
        vec![true, true, false]
    );
}

#[test]
fn test_int_reduce() {
    let t = int_tensor(vec![1, 2, 3, 4, 5, 6], [2, 3]);

    assert_eq!(read_i32(GnaBackend::int_sum(t.clone())), vec![21]);
    assert_eq!(
        read_i32(GnaBackend::int_sum_dim(t.clone(), 0)),
        vec![5, 7, 9]
    );
    assert_eq!(read_i32(GnaBackend::int_sum_dim(t.clone(), 1)), vec![6, 15]);
    assert_eq!(read_i32(GnaBackend::int_prod(t.clone())), vec![720]);
    assert_eq!(
        read_i32(GnaBackend::int_prod_dim(t.clone(), 0)),
        vec![4, 10, 18]
    );
    assert_eq!(read_i32(GnaBackend::int_mean_dim(t.clone(), 1)), vec![2, 5]);

    assert_eq!(
        read_i32(GnaBackend::int_cumsum(t.clone(), 1)),
        vec![1, 3, 6, 4, 9, 15]
    );
    assert_eq!(
        read_i32(GnaBackend::int_cumprod(t.clone(), 1)),
        vec![1, 2, 6, 4, 20, 120]
    );
    assert_eq!(
        read_i32(GnaBackend::int_cummin(t.clone(), 1)),
        vec![1, 1, 1, 4, 4, 4]
    );
    assert_eq!(
        read_i32(GnaBackend::int_cummax(t.clone(), 1)),
        vec![1, 2, 3, 4, 5, 6]
    );

    let arg_t = int_tensor(vec![3, 1, 5, 2, 8, 4], [2, 3]);
    assert_eq!(
        read_i32(GnaBackend::int_argmax(arg_t.clone(), 1)),
        vec![2, 1]
    );
    assert_eq!(
        read_i32(GnaBackend::int_argmin(arg_t.clone(), 1)),
        vec![1, 0]
    );
    assert_eq!(
        read_i32(GnaBackend::int_argtopk(arg_t, 1, 2)),
        vec![2, 0, 1, 2]
    );
}

#[test]
fn test_int_indexing() {
    let t = int_tensor(vec![1, 2, 3, 4, 5, 6], [2, 3]);

    let sliced = GnaBackend::int_slice(t.clone(), &[Slice::from(0..2), Slice::from(1..3)]);
    assert_eq!(sliced.shape(), Shape::new([2, 2]));
    assert_eq!(read_i32(sliced), vec![2, 3, 5, 6]);

    let assigned = GnaBackend::int_slice_assign(
        t.clone(),
        &[Slice::from(0..1), Slice::from(0..2)],
        int_tensor(vec![10, 20], [1, 2]),
    );
    assert_eq!(read_i32(assigned), vec![10, 20, 3, 4, 5, 6]);

    let mask = GnaBackend::bool_from_data(
        TensorData::new(vec![true, false, true, false, true, false], [2, 3]),
        &dev(),
    );
    let where_t = GnaBackend::int_mask_where(
        t.clone(),
        mask.clone(),
        int_tensor(vec![9, 9, 9, 9, 9, 9], [2, 3]),
    );
    assert_eq!(read_i32(where_t), vec![9, 2, 9, 4, 9, 6]);

    let fill_t = GnaBackend::int_mask_fill(t.clone(), mask, s(-1));
    assert_eq!(read_i32(fill_t), vec![-1, 2, -1, 4, -1, 6]);

    let gathered = GnaBackend::int_gather(1, t.clone(), int_tensor(vec![2, 0, 1, 2], [2, 2]));
    assert_eq!(read_i32(gathered), vec![3, 1, 5, 6]);

    let scattered = GnaBackend::int_scatter(
        1,
        t.clone(),
        int_tensor(vec![0, 2], [2, 1]),
        int_tensor(vec![10, 20], [2, 1]),
        IndexingUpdateOp::Add,
    );
    assert_eq!(read_i32(scattered), vec![11, 2, 3, 4, 5, 26]);

    let selected = GnaBackend::int_select(t.clone(), 1, int_tensor_1d(vec![2, 0]));
    assert_eq!(selected.shape(), Shape::new([2, 2]));
    assert_eq!(read_i32(selected), vec![3, 1, 6, 4]);

    let sel_assigned = GnaBackend::int_select_assign(
        t.clone(),
        1,
        int_tensor_1d(vec![0]),
        int_tensor(vec![10, 20], [2, 1]),
        IndexingUpdateOp::Add,
    );
    assert_eq!(read_i32(sel_assigned), vec![11, 2, 3, 24, 5, 6]);

    let swapped = GnaBackend::int_swap_dims(t.clone(), 0, 1);
    assert_eq!(swapped.shape(), Shape::new([3, 2]));
    assert_eq!(read_i32(swapped), vec![1, 4, 2, 5, 3, 6]);

    let permuted = GnaBackend::int_permute(t.clone(), &[1, 0]);
    assert_eq!(permuted.shape(), Shape::new([3, 2]));
    assert_eq!(read_i32(permuted), vec![1, 4, 2, 5, 3, 6]);

    let flipped = GnaBackend::int_flip(t, &[1]);
    assert_eq!(read_i32(flipped), vec![3, 2, 1, 6, 5, 4]);

    let unfolded = GnaBackend::int_unfold(int_tensor_1d(vec![1, 2, 3, 4]), 0, 2, 1);
    assert_eq!(unfolded.shape(), Shape::new([3, 2]));
    assert_eq!(read_i32(unfolded), vec![1, 2, 2, 3, 3, 4]);
}

#[test]
fn test_int_dtypes_from_data_and_cast() {
    let t_i8 = GnaBackend::int_from_data(TensorData::new(vec![1i8, -2, 3], [3]), &dev());
    assert_eq!(t_i8.dtype(), IntDType::I8);
    let t_i16 = GnaBackend::int_from_data(TensorData::new(vec![10i16, -20, 30], [3]), &dev());
    assert_eq!(t_i16.dtype(), IntDType::I16);
    let t_u8 = GnaBackend::int_from_data(TensorData::new(vec![1u8, 2, 3], [3]), &dev());
    assert_eq!(t_u8.dtype(), IntDType::U8);
    let t_u16 = GnaBackend::int_from_data(TensorData::new(vec![10u16, 20, 30], [3]), &dev());
    assert_eq!(t_u16.dtype(), IntDType::U16);
    let t_u32 = GnaBackend::int_from_data(TensorData::new(vec![100u32, 200, 300], [3]), &dev());
    assert_eq!(t_u32.dtype(), IntDType::U32);
    let t_u64 = GnaBackend::int_from_data(TensorData::new(vec![1000u64, 2000, 3000], [3]), &dev());
    assert_eq!(t_u64.dtype(), IntDType::U64);

    let base = int_tensor_1d(vec![1, 2, 3]);
    for dt in [
        IntDType::I8,
        IntDType::I16,
        IntDType::I64,
        IntDType::U8,
        IntDType::U16,
        IntDType::U32,
        IntDType::U64,
    ] {
        let casted = GnaBackend::int_cast(base.clone(), dt);
        assert_eq!(casted.dtype(), dt);
        let back = GnaBackend::int_cast(casted, IntDType::I32);
        assert_eq!(read_i32(back), vec![1, 2, 3]);
    }
}

#[test]
fn test_int_scatter_select_assign_ops() {
    let base = int_tensor(vec![2, 4, 6, 8, 10, 12], [2, 3]);
    let idx = int_tensor(vec![0, 2], [2, 1]);
    let val = int_tensor(vec![3, 5], [2, 1]);

    for (op, expected) in [
        (IndexingUpdateOp::Add, vec![5, 4, 6, 8, 10, 17]),
        (IndexingUpdateOp::Mul, vec![6, 4, 6, 8, 10, 60]),
        (IndexingUpdateOp::Min, vec![2, 4, 6, 8, 10, 5]),
        (IndexingUpdateOp::Max, vec![3, 4, 6, 8, 10, 12]),
    ] {
        let out = GnaBackend::int_scatter(1, base.clone(), idx.clone(), val.clone(), op);
        assert_eq!(read_i32(out), expected);
    }

    let sel_idx = int_tensor_1d(vec![1]);
    let sel_val = int_tensor(vec![3, 20], [2, 1]);
    for (op, expected) in [
        (IndexingUpdateOp::Add, vec![2, 7, 6, 8, 30, 12]),
        (IndexingUpdateOp::Mul, vec![2, 12, 6, 8, 200, 12]),
        (IndexingUpdateOp::Min, vec![2, 3, 6, 8, 10, 12]),
        (IndexingUpdateOp::Max, vec![2, 4, 6, 8, 20, 12]),
    ] {
        let out = GnaBackend::int_select_assign(base.clone(), 1, sel_idx.clone(), sel_val.clone(), op);
        assert_eq!(read_i32(out), expected);
    }
}
