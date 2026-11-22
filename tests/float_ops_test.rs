use burn_backend::ops::{BoolTensorOps, FloatTensorOps, IntTensorOps, ModuleOps};
use burn_backend::{
    BoolDType, DType, Distribution, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
    TensorMetadata, tensor::IndexingUpdateOp, try_read_sync,
};
use nanai_burn_gna::{GnaBackend, GnaBuffer, GnaDevice, GnaFloatTensorPrimitive};

fn device() -> GnaDevice {
    GnaDevice::default()
}

fn make_f32(data: TensorData) -> GnaFloatTensorPrimitive {
    GnaBackend::float_from_data(data, &device())
}

fn read_f32(tensor: GnaFloatTensorPrimitive) -> Vec<f32> {
    let data = try_read_sync(GnaBackend::float_into_data(tensor))
        .expect("sync read")
        .expect("float_into_data");
    data.try_to_vec::<f32>().expect("to_vec f32")
}

fn read_i32(tensor: nanai_burn_gna::GnaIntTensorPrimitive) -> Vec<i32> {
    let data = try_read_sync(GnaBackend::int_into_data(tensor))
        .expect("sync read")
        .expect("int_into_data");
    data.try_to_vec::<i32>().expect("to_vec i32")
}

fn read_bool(tensor: nanai_burn_gna::GnaBoolTensorPrimitive) -> Vec<bool> {
    let data = try_read_sync(GnaBackend::bool_into_data(tensor))
        .expect("sync read")
        .expect("bool_into_data");
    data.try_to_vec::<bool>().expect("to_vec bool")
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: actual {:?} vs expected {:?}",
        actual,
        expected
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "mismatch at index {i}: actual={a}, expected={e}, tol={tol}"
        );
    }
}

#[test]
fn test_gna_float_tensor_primitive_methods() {
    let buf = GnaBuffer::zeros(16);
    let prim = GnaFloatTensorPrimitive::new(Shape::new([2, 2]), FloatDType::F32, buf);

    assert_eq!(prim.dtype(), FloatDType::F32);
    assert_eq!(TensorMetadata::dtype(&prim), DType::F32);
    assert_eq!(prim.shape().dims(), [2, 2]);
    assert_eq!(prim.device(), GnaDevice::default());
    assert!(prim.can_mut());
    assert_eq!(prim.buffer().len(), 16);

    let reshaped = prim.clone().with_shape(Shape::new([4, 1]));
    assert_eq!(reshaped.shape().dims(), [4, 1]);

    let retyped = reshaped.with_dtype(FloatDType::F16);
    assert_eq!(retyped.dtype(), FloatDType::F16);
    assert_eq!(TensorMetadata::dtype(&retyped), DType::F16);

    let (shape, dtype, buf2) = retyped.into_parts();
    assert_eq!(shape.dims(), [4, 1]);
    assert_eq!(dtype, FloatDType::F16);
    assert_eq!(buf2.len(), 16);

    let buf3 = prim.into_buffer();
    assert_eq!(buf3.len(), 16);
}

#[test]
fn test_float_creation_ops() {
    let dev = device();

    // empty
    let empty = GnaBackend::float_empty(Shape::new([2, 3]), &dev, FloatDType::F32);
    assert_eq!(empty.shape().dims(), [2, 3]);
    assert_eq!(empty.dtype(), FloatDType::F32);

    // to_device
    let t = make_f32(TensorData::from([1.0f32, 2.0, 3.0, 4.0]));
    let t_dev = GnaBackend::float_to_device(t.clone(), &dev);
    assert_close(&read_f32(t_dev), &[1.0, 2.0, 3.0, 4.0], 1e-6);

    // into_int
    let i = GnaBackend::float_into_int(t.clone(), IntDType::I32);
    assert_eq!(read_i32(i), vec![1, 2, 3, 4]);

    // cast
    let cast_f16 = GnaBackend::float_cast(t.clone(), FloatDType::F16);
    assert_eq!(cast_f16.dtype(), FloatDType::F16);
    let cast_f32 = GnaBackend::float_cast(cast_f16, FloatDType::F32);
    assert_close(&read_f32(cast_f32), &[1.0, 2.0, 3.0, 4.0], 1e-3);

    // reshape
    let reshaped = GnaBackend::float_reshape(t.clone(), Shape::new([2, 2]));
    assert_eq!(reshaped.shape().dims(), [2, 2]);
    assert_close(&read_f32(reshaped), &[1.0, 2.0, 3.0, 4.0], 1e-6);

    // expand
    let row = make_f32(TensorData::from([[1.0f32, 2.0]]));
    let expanded = GnaBackend::float_expand(row, Shape::new([3, 2]));
    assert_eq!(expanded.shape().dims(), [3, 2]);
    assert_close(&read_f32(expanded), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0], 1e-6);

    // random
    let rand_u = GnaBackend::float_random(
        Shape::new([16]),
        Distribution::Uniform(0.0, 1.0),
        &dev,
        FloatDType::F32,
    );
    let vals_u = read_f32(rand_u);
    assert_eq!(vals_u.len(), 16);
    assert!(vals_u.iter().all(|&v| (0.0..=1.0).contains(&v)));

    let rand_b = GnaBackend::float_random(
        Shape::new([16]),
        Distribution::Bernoulli(0.5),
        &dev,
        FloatDType::F32,
    );
    let vals_b = read_f32(rand_b);
    assert!(vals_b.iter().all(|&v| v == 0.0 || v == 1.0));
}

#[test]
fn test_float_arithmetic_ops() {
    let a = make_f32(TensorData::from([2.0f32, 4.0, 6.0, 8.0]));
    let b = make_f32(TensorData::from([1.0f32, 2.0, 3.0, 4.0]));

    // GNA hardware ops: add, sub, mul, matmul
    assert_close(
        &read_f32(GnaBackend::float_add(a.clone(), b.clone())),
        &[3.0, 6.0, 9.0, 12.0],
        0.05,
    );
    assert_close(
        &read_f32(GnaBackend::float_add_scalar(
            a.clone(),
            Scalar::from(1.0f32),
        )),
        &[3.0, 5.0, 7.0, 9.0],
        0.05,
    );
    assert_close(
        &read_f32(GnaBackend::float_sub(a.clone(), b.clone())),
        &[1.0, 2.0, 3.0, 4.0],
        0.05,
    );
    assert_close(
        &read_f32(GnaBackend::float_sub_scalar(
            a.clone(),
            Scalar::from(1.0f32),
        )),
        &[1.0, 3.0, 5.0, 7.0],
        0.05,
    );
    assert_close(
        &read_f32(GnaBackend::float_mul(a.clone(), b.clone())),
        &[2.0, 8.0, 18.0, 32.0],
        0.1,
    );
    assert_close(
        &read_f32(GnaBackend::float_mul_scalar(
            a.clone(),
            Scalar::from(2.0f32),
        )),
        &[4.0, 8.0, 12.0, 16.0],
        0.1,
    );

    // div, remainder, recip, powf, atan2
    assert_close(
        &read_f32(GnaBackend::float_div(a.clone(), b.clone())),
        &[2.0, 2.0, 2.0, 2.0],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_div_scalar(
            a.clone(),
            Scalar::from(2.0f32),
        )),
        &[1.0, 2.0, 3.0, 4.0],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_remainder(a.clone(), b.clone())),
        &[0.0, 0.0, 0.0, 0.0],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_remainder_scalar(
            a.clone(),
            Scalar::from(5.0f32),
        )),
        &[2.0, 4.0, 1.0, 3.0],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_recip(b.clone())),
        &[1.0, 0.5, 1.0 / 3.0, 0.25],
        1e-5,
    );

    let base = make_f32(TensorData::from([2.0f32, 3.0]));
    let exp = make_f32(TensorData::from([3.0f32, 2.0]));
    assert_close(
        &read_f32(GnaBackend::float_powf(base.clone(), exp)),
        &[8.0, 9.0],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_powf_scalar_impl(
            base,
            Scalar::from(2.0f32),
        )),
        &[4.0, 9.0],
        1e-5,
    );

    let y = make_f32(TensorData::from([1.0f32, 0.0]));
    let x = make_f32(TensorData::from([1.0f32, 1.0]));
    assert_close(
        &read_f32(GnaBackend::float_atan2(y, x)),
        &[std::f32::consts::FRAC_PI_4, 0.0],
        1e-5,
    );

    // cross
    let c1 = make_f32(TensorData::from([[1.0f32, 0.0, 0.0]]));
    let c2 = make_f32(TensorData::from([[0.0f32, 1.0, 0.0]]));
    assert_close(
        &read_f32(GnaBackend::float_cross(c1, c2, 1)),
        &[0.0, 0.0, 1.0],
        1e-5,
    );

    // matmul (GNA hardware)
    let m1 = make_f32(TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]));
    let m2 = make_f32(TensorData::from([[2.0f32, 0.0], [1.0, 2.0]]));
    assert_close(
        &read_f32(GnaBackend::float_matmul(m1, m2)),
        &[4.0, 4.0, 10.0, 8.0],
        0.1,
    );
}

#[test]
fn test_float_comparison_ops() {
    let a = make_f32(TensorData::from([1.0f32, 2.0, 3.0]));
    let b = make_f32(TensorData::from([2.0f32, 2.0, 1.0]));

    assert_eq!(
        read_bool(GnaBackend::float_equal(
            a.clone(),
            b.clone(),
            BoolDType::Native
        )),
        vec![false, true, false]
    );
    assert_eq!(
        read_bool(GnaBackend::float_equal_elem(
            a.clone(),
            Scalar::from(2.0f32),
            BoolDType::Native
        )),
        vec![false, true, false]
    );

    assert_eq!(
        read_bool(GnaBackend::float_greater(
            a.clone(),
            b.clone(),
            BoolDType::Native
        )),
        vec![false, false, true]
    );
    assert_eq!(
        read_bool(GnaBackend::float_greater_elem(
            a.clone(),
            Scalar::from(2.0f32),
            BoolDType::Native
        )),
        vec![false, false, true]
    );

    assert_eq!(
        read_bool(GnaBackend::float_greater_equal(
            a.clone(),
            b.clone(),
            BoolDType::Native
        )),
        vec![false, true, true]
    );
    assert_eq!(
        read_bool(GnaBackend::float_greater_equal_elem(
            a.clone(),
            Scalar::from(2.0f32),
            BoolDType::Native
        )),
        vec![false, true, true]
    );

    assert_eq!(
        read_bool(GnaBackend::float_lower(
            a.clone(),
            b.clone(),
            BoolDType::Native
        )),
        vec![true, false, false]
    );
    assert_eq!(
        read_bool(GnaBackend::float_lower_elem(
            a.clone(),
            Scalar::from(2.0f32),
            BoolDType::Native
        )),
        vec![true, false, false]
    );

    assert_eq!(
        read_bool(GnaBackend::float_lower_equal(
            a.clone(),
            b.clone(),
            BoolDType::Native
        )),
        vec![true, true, false]
    );
    assert_eq!(
        read_bool(GnaBackend::float_lower_equal_elem(
            a,
            Scalar::from(2.0f32),
            BoolDType::Native
        )),
        vec![true, true, false]
    );
}

#[test]
fn test_float_indexing_ops() {
    let dev = device();
    let t = make_f32(TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]));

    // swap_dims
    let swapped = GnaBackend::float_swap_dims(t.clone(), 0, 1);
    assert_eq!(swapped.shape().dims(), [3, 2]);
    assert_close(&read_f32(swapped), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-6);

    // permute
    let permuted = GnaBackend::float_permute(t.clone(), &[1, 0]);
    assert_eq!(permuted.shape().dims(), [3, 2]);
    assert_close(&read_f32(permuted), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-6);

    // flip
    let flipped = GnaBackend::float_flip(t.clone(), &[0]);
    assert_close(&read_f32(flipped), &[4.0, 5.0, 6.0, 1.0, 2.0, 3.0], 1e-6);

    // gather & scatter
    let idx = GnaBackend::int_from_data(TensorData::from([[2i32, 0], [1, 2]]), &dev);
    let gathered = GnaBackend::float_gather(1, t.clone(), idx.clone());
    assert_close(&read_f32(gathered), &[3.0, 1.0, 5.0, 6.0], 1e-6);

    let zeros = make_f32(TensorData::from([[0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0]]));
    let src = make_f32(TensorData::from([[10.0f32, 20.0], [30.0, 40.0]]));
    let scattered = GnaBackend::float_scatter(1, zeros, idx, src, IndexingUpdateOp::Add);
    assert_close(
        &read_f32(scattered),
        &[20.0, 0.0, 10.0, 0.0, 30.0, 40.0],
        1e-6,
    );

    // select & select_assign
    let sel_idx = GnaBackend::int_from_data(TensorData::from([1i32, 0]), &dev);
    let selected = GnaBackend::float_select(t.clone(), 0, sel_idx.clone());
    assert_close(&read_f32(selected), &[4.0, 5.0, 6.0, 1.0, 2.0, 3.0], 1e-6);

    let add_val = make_f32(TensorData::from([[1.0f32, 1.0, 1.0], [2.0, 2.0, 2.0]]));
    let sel_assigned =
        GnaBackend::float_select_assign(t.clone(), 0, sel_idx, add_val, IndexingUpdateOp::Add);
    assert_close(
        &read_f32(sel_assigned),
        &[3.0, 4.0, 5.0, 5.0, 6.0, 7.0],
        1e-6,
    );

    // slice & slice_assign
    let slices = [Slice::new(0, Some(1), 1), Slice::new(1, Some(3), 1)];
    let sliced = GnaBackend::float_slice(t.clone(), &slices);
    assert_eq!(sliced.shape().dims(), [1, 2]);
    assert_close(&read_f32(sliced), &[2.0, 3.0], 1e-6);

    let patch = make_f32(TensorData::from([[9.0f32, 8.0]]));
    let slice_assigned = GnaBackend::float_slice_assign(t.clone(), &slices, patch);
    assert_close(
        &read_f32(slice_assigned),
        &[1.0, 9.0, 8.0, 4.0, 5.0, 6.0],
        1e-6,
    );

    // mask_where & mask_fill
    let mask = GnaBackend::bool_from_data(
        TensorData::from([[true, false, true], [false, true, false]]),
        &dev,
    );
    let fill_val = make_f32(TensorData::from([[0.0f32, 0.0, 0.0], [0.0, 0.0, 0.0]]));
    let masked_where = GnaBackend::float_mask_where(t.clone(), mask.clone(), fill_val);
    assert_close(
        &read_f32(masked_where),
        &[0.0, 2.0, 0.0, 4.0, 0.0, 6.0],
        1e-6,
    );

    let masked_fill = GnaBackend::float_mask_fill(t.clone(), mask, Scalar::from(-1.0f32));
    assert_close(
        &read_f32(masked_fill),
        &[-1.0, 2.0, -1.0, 4.0, -1.0, 6.0],
        1e-6,
    );

    // unfold
    let vec_t = make_f32(TensorData::from([1.0f32, 2.0, 3.0, 4.0]));
    let unfolded = GnaBackend::float_unfold(vec_t, 0, 2, 1);
    assert_eq!(unfolded.shape().dims(), [3, 2]);
    assert_close(&read_f32(unfolded), &[1.0, 2.0, 2.0, 3.0, 3.0, 4.0], 1e-6);
}

#[test]
fn test_float_reduce_ops() {
    let t = make_f32(TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]));

    // GNA hardware sum / sum_dim / mean_dim
    assert_close(&read_f32(GnaBackend::float_sum(t.clone())), &[21.0], 0.1);
    assert_close(
        &read_f32(GnaBackend::float_sum_dim(t.clone(), 1)),
        &[6.0, 15.0],
        0.1,
    );
    assert_close(
        &read_f32(GnaBackend::float_mean_dim(t.clone(), 1)),
        &[2.0, 5.0],
        0.1,
    );

    // cumsum, cumprod, cummin, cummax
    assert_close(
        &read_f32(GnaBackend::float_cumsum(t.clone(), 1)),
        &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_cumprod(t.clone(), 1)),
        &[1.0, 2.0, 6.0, 4.0, 20.0, 120.0],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_cummin(t.clone(), 1)),
        &[1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_cummax(t.clone(), 1)),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        1e-5,
    );

    // argmax, argmin, argtopk
    assert_eq!(
        read_i32(GnaBackend::float_argmax(t.clone(), 1, IntDType::I32)),
        vec![2, 2]
    );
    assert_eq!(
        read_i32(GnaBackend::float_argmin(t.clone(), 1, IntDType::I32)),
        vec![0, 0]
    );
    assert_eq!(
        read_i32(GnaBackend::float_argtopk(t, 1, 2, IntDType::I32)),
        vec![2, 1, 2, 1]
    );
}

#[test]
fn test_float_unary_ops() {
    let pos = make_f32(TensorData::from([1.0f32, 4.0]));
    assert_close(
        &read_f32(GnaBackend::float_exp(pos.clone())),
        &[1.0f32.exp(), 4.0f32.exp()],
        1e-4,
    );
    assert_close(
        &read_f32(GnaBackend::float_log(pos.clone())),
        &[0.0, 4.0f32.ln()],
        1e-4,
    );
    assert_close(
        &read_f32(GnaBackend::float_log1p(pos.clone())),
        &[2.0f32.ln(), 5.0f32.ln()],
        1e-4,
    );
    assert_close(&read_f32(GnaBackend::float_sqrt(pos)), &[1.0, 2.0], 1e-5);

    let signed = make_f32(TensorData::from([-1.5f32, 2.3]));
    assert_close(
        &read_f32(GnaBackend::float_abs(signed.clone())),
        &[1.5, 2.3],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_round(signed.clone())),
        &[-2.0, 2.0],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_floor(signed.clone())),
        &[-2.0, 2.0],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_ceil(signed.clone())),
        &[-1.0, 3.0],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_trunc(signed)),
        &[-1.0, 2.0],
        1e-5,
    );

    let unit = make_f32(TensorData::from([0.0f32, 0.5]));
    assert_close(
        &read_f32(GnaBackend::float_cos(unit.clone())),
        &[1.0, 0.5f32.cos()],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_sin(unit.clone())),
        &[0.0, 0.5f32.sin()],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_tan(unit.clone())),
        &[0.0, 0.5f32.tan()],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_cosh(unit.clone())),
        &[1.0, 0.5f32.cosh()],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_sinh(unit.clone())),
        &[0.0, 0.5f32.sinh()],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_tanh(unit.clone())),
        &[0.0, 0.5f32.tanh()],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_acos(unit.clone())),
        &[0.0f32.acos(), 0.5f32.acos()],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_asin(unit.clone())),
        &[0.0, 0.5f32.asin()],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_asinh(unit.clone())),
        &[0.0, 0.5f32.asinh()],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_atan(unit.clone())),
        &[0.0, 0.5f32.atan()],
        1e-5,
    );
    assert_close(
        &read_f32(GnaBackend::float_atanh(unit.clone())),
        &[0.0, 0.5f32.atanh()],
        1e-5,
    );

    let ge1 = make_f32(TensorData::from([1.0f32, 2.0]));
    assert_close(
        &read_f32(GnaBackend::float_acosh(ge1)),
        &[0.0, 2.0f32.acosh()],
        1e-5,
    );

    let erf_t = make_f32(TensorData::from([0.0f32]));
    assert_close(&read_f32(GnaBackend::float_erf(erf_t)), &[0.0], 1e-5);
}

#[test]
fn test_float_indexing_update_ops() {
    let dev = device();
    let base = make_f32(TensorData::from([[2.0f32, 4.0, 6.0], [8.0, 10.0, 12.0]]));
    let idx = GnaBackend::int_from_data(TensorData::from([[0i32, 2], [1, 0]]), &dev);
    let src = make_f32(TensorData::from([[3.0f32, 5.0], [2.0, 4.0]]));

    // float_scatter with Add, Mul, Min, Max
    let sc_add = GnaBackend::float_scatter(
        1,
        base.clone(),
        idx.clone(),
        src.clone(),
        IndexingUpdateOp::Add,
    );
    assert_close(&read_f32(sc_add), &[5.0, 4.0, 11.0, 12.0, 12.0, 12.0], 1e-5);

    let sc_mul = GnaBackend::float_scatter(
        1,
        base.clone(),
        idx.clone(),
        src.clone(),
        IndexingUpdateOp::Mul,
    );
    assert_close(&read_f32(sc_mul), &[6.0, 4.0, 30.0, 32.0, 20.0, 12.0], 1e-5);

    let sc_min = GnaBackend::float_scatter(
        1,
        base.clone(),
        idx.clone(),
        src.clone(),
        IndexingUpdateOp::Min,
    );
    assert_close(&read_f32(sc_min), &[2.0, 4.0, 5.0, 4.0, 2.0, 12.0], 1e-5);

    let sc_max = GnaBackend::float_scatter(1, base.clone(), idx, src, IndexingUpdateOp::Max);
    assert_close(&read_f32(sc_max), &[3.0, 4.0, 6.0, 8.0, 10.0, 12.0], 1e-5);

    // float_select_assign with Add, Mul, Min, Max
    let sel_idx = GnaBackend::int_from_data(TensorData::from([1i32, 0]), &dev);
    let sel_val = make_f32(TensorData::from([[2.0f32, 3.0, 4.0], [5.0, 6.0, 7.0]]));

    let sa_add = GnaBackend::float_select_assign(
        base.clone(),
        0,
        sel_idx.clone(),
        sel_val.clone(),
        IndexingUpdateOp::Add,
    );
    assert_close(
        &read_f32(sa_add),
        &[7.0, 10.0, 13.0, 10.0, 13.0, 16.0],
        1e-5,
    );

    let sa_mul = GnaBackend::float_select_assign(
        base.clone(),
        0,
        sel_idx.clone(),
        sel_val.clone(),
        IndexingUpdateOp::Mul,
    );
    assert_close(
        &read_f32(sa_mul),
        &[10.0, 24.0, 42.0, 16.0, 30.0, 48.0],
        1e-5,
    );

    let sa_min = GnaBackend::float_select_assign(
        base.clone(),
        0,
        sel_idx.clone(),
        sel_val.clone(),
        IndexingUpdateOp::Min,
    );
    assert_close(&read_f32(sa_min), &[2.0, 4.0, 6.0, 2.0, 3.0, 4.0], 1e-5);

    let sa_max = GnaBackend::float_select_assign(base, 0, sel_idx, sel_val, IndexingUpdateOp::Max);
    assert_close(&read_f32(sa_max), &[5.0, 6.0, 7.0, 8.0, 10.0, 12.0], 1e-5);
}

#[test]
fn test_float_dtypes_f16_bf16_f64() {
    let dev = device();

    // F16
    let f16_data = TensorData::from([1.0f32, 2.0, 3.0]).convert_dtype(DType::F16);
    let t_f16 = GnaBackend::float_from_data(f16_data, &dev);
    assert_eq!(t_f16.dtype(), FloatDType::F16);
    let f16_to_f32 = GnaBackend::float_cast(t_f16, FloatDType::F32);
    assert_close(&read_f32(f16_to_f32), &[1.0, 2.0, 3.0], 1e-3);

    // BF16
    let bf16_data = TensorData::from([1.0f32, 2.0, 3.0]).convert_dtype(DType::BF16);
    let t_bf16 = GnaBackend::float_from_data(bf16_data, &dev);
    assert_eq!(t_bf16.dtype(), FloatDType::BF16);
    let bf16_to_f32 = GnaBackend::float_cast(t_bf16, FloatDType::F32);
    assert_close(&read_f32(bf16_to_f32), &[1.0, 2.0, 3.0], 1e-2);

    // F64
    let f64_data = TensorData::from([1.5f64, 2.5, 3.5]);
    let t_f64 = GnaBackend::float_from_data(f64_data, &dev);
    assert_eq!(t_f64.dtype(), FloatDType::F64);
    let f64_to_f32 = GnaBackend::float_cast(t_f64, FloatDType::F32);
    assert_close(&read_f32(f64_to_f32), &[1.5, 2.5, 3.5], 1e-6);
}

#[test]
fn test_avg_pool2d_options() {
    // Input: [1, 1, 3, 3]
    let x = make_f32(TensorData::from([[[
        [1.0f32, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]]]));

    // count_include_pad = true, ceil_mode = false, padding = [1, 1], kernel = [2, 2], stride = [2, 2]
    let out_pad = GnaBackend::avg_pool2d(x.clone(), [2, 2], [2, 2], [1, 1], true, false);
    assert_eq!(out_pad.shape().dims(), [1, 1, 2, 2]);
    assert_close(&read_f32(out_pad), &[0.25, 1.25, 2.75, 7.0], 0.1);

    // count_include_pad = false, ceil_mode = true, padding = [0, 0], kernel = [2, 2], stride = [2, 2]
    let out_ceil = GnaBackend::avg_pool2d(x, [2, 2], [2, 2], [0, 0], false, true);
    assert_eq!(out_ceil.shape().dims(), [1, 1, 2, 2]);
    assert_close(&read_f32(out_ceil), &[3.0, 4.5, 7.5, 9.0], 0.1);
}

#[test]
fn test_float_matmul_zero_tensors() {
    let zeros = make_f32(TensorData::from([[0.0f32, 0.0], [0.0, 0.0]]));
    let nonzero = make_f32(TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]));

    let out1 = GnaBackend::float_matmul(zeros.clone(), nonzero.clone());
    assert_close(&read_f32(out1), &[0.0, 0.0, 0.0, 0.0], 1e-6);

    let out2 = GnaBackend::float_matmul(nonzero, zeros);
    assert_close(&read_f32(out2), &[0.0, 0.0, 0.0, 0.0], 1e-6);
}

#[test]
fn test_float_mask_fill_f16_and_f64() {
    let dev = device();
    let mask = GnaBackend::bool_from_data(TensorData::from([true, false, true]), &dev);

    // F16 (2-byte scalar_to_le_bytes)
    let f16_t = GnaBackend::float_from_data(
        TensorData::from([1.0f32, 2.0, 3.0]).convert_dtype(DType::F16),
        &dev,
    );
    let filled_f16 = GnaBackend::float_mask_fill(f16_t, mask.clone(), Scalar::from(0.0f32));
    assert_eq!(filled_f16.dtype(), FloatDType::F16);
    assert_close(
        &read_f32(GnaBackend::float_cast(filled_f16, FloatDType::F32)),
        &[0.0, 2.0, 0.0],
        1e-3,
    );

    // F64 (8-byte scalar_to_le_bytes)
    let f64_t = GnaBackend::float_from_data(TensorData::from([1.0f64, 2.0, 3.0]), &dev);
    let filled_f64 = GnaBackend::float_mask_fill(f64_t, mask, Scalar::from(-5.0f64));
    assert_eq!(filled_f64.dtype(), FloatDType::F64);
    assert_close(
        &read_f32(GnaBackend::float_cast(filled_f64, FloatDType::F32)),
        &[-5.0, 2.0, -5.0],
        1e-6,
    );
}

#[test]
#[should_panic]
fn test_float_matmul_1d_rhs_panics() {
    let lhs = make_f32(TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]));
    let rhs = make_f32(TensorData::from([1.0f32, 2.0]));
    let _ = GnaBackend::float_matmul(lhs, rhs);
}
