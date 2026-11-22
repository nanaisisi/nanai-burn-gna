use burn_backend::ops::{BoolTensorOps, FloatTensorOps, IntTensorOps, QTensorOps};
use burn_backend::quantization::QuantizationParametersPrimitive;
use burn_backend::{
    BoolStore, DType, FloatDType, IntDType, Scalar, Shape, Slice, TensorData, TensorMetadata,
    try_read_sync,
};
use nanai_burn_gna::{
    GnaBackend, GnaBoolTensorPrimitive, GnaBuffer, GnaDevice, GnaQTensorPrimitive,
};

fn read_bool(t: GnaBoolTensorPrimitive) -> Vec<bool> {
    let data = try_read_sync(GnaBackend::bool_into_data(t))
        .expect("sync read")
        .expect("bool_into_data");
    data.as_slice::<bool>().expect("bool slice").to_vec()
}

fn read_f32(t: nanai_burn_gna::GnaFloatTensorPrimitive) -> Vec<f32> {
    let data = try_read_sync(GnaBackend::float_into_data(t))
        .expect("sync read")
        .expect("float_into_data");
    data.as_slice::<f32>().expect("f32 slice").to_vec()
}

fn read_i32(t: nanai_burn_gna::GnaIntTensorPrimitive) -> Vec<i32> {
    let data = try_read_sync(GnaBackend::int_into_data(t))
        .expect("sync read")
        .expect("int_into_data");
    data.as_slice::<i32>().expect("i32 slice").to_vec()
}

fn bool_tensor(vals: &[bool], dims: &[usize]) -> GnaBoolTensorPrimitive {
    let device = GnaDevice::default();
    let data = TensorData::new(vals.to_vec(), Shape::from(dims));
    GnaBackend::bool_from_data(data, &device)
}

fn i32_tensor(vals: &[i32], dims: &[usize]) -> nanai_burn_gna::GnaIntTensorPrimitive {
    let device = GnaDevice::default();
    let data = TensorData::new(vals.to_vec(), Shape::from(dims));
    GnaBackend::int_from_data(data, &device)
}

// ---------------------------------------------------------------------------
// GnaBuffer tests
// ---------------------------------------------------------------------------

#[test]
fn gna_buffer_methods() {
    let empty = GnaBuffer::zeros(0);
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.as_slice(), &[] as &[u8]);

    let zeros = GnaBuffer::zeros(4);
    assert!(!zeros.is_empty());
    assert_eq!(zeros.len(), 4);
    assert_eq!(zeros.as_slice(), &[0, 0, 0, 0]);

    let filled = GnaBuffer::filled(7, 3);
    assert_eq!(filled.to_vec(), vec![7, 7, 7]);
    assert!(filled.can_mut());

    let cloned = filled.clone();
    assert!(!filled.can_mut());
    assert!(!cloned.can_mut());
    drop(cloned);
    assert!(filled.can_mut());
    assert_eq!(filled.into_vec(), vec![7, 7, 7]);

    let from_b = GnaBuffer::from_bytes(vec![10, 20, 30]);
    assert_eq!(from_b.as_slice(), &[10, 20, 30]);
}

// ---------------------------------------------------------------------------
// GnaBoolTensorPrimitive & GnaQTensorPrimitive tests
// ---------------------------------------------------------------------------

#[test]
fn gna_bool_primitive_methods() {
    let buf = GnaBuffer::from_bytes(vec![1, 0, 1, 0]);
    let prim = GnaBoolTensorPrimitive::new(Shape::from([2, 2]), buf);

    assert_eq!(prim.dtype(), DType::Bool(BoolStore::Native));
    assert_eq!(prim.shape(), Shape::from([2, 2]));
    assert_eq!(prim.device(), GnaDevice::default());
    assert!(prim.can_mut());
    assert_eq!(prim.buffer().as_slice(), &[1, 0, 1, 0]);

    let reshaped = prim.with_shape(Shape::from([4]));
    assert_eq!(reshaped.shape(), Shape::from([4]));

    let (shape, buf2) = reshaped.clone().into_parts();
    assert_eq!(shape, Shape::from([4]));
    assert_eq!(buf2.as_slice(), &[1, 0, 1, 0]);

    let buf3 = reshaped.into_buffer();
    assert_eq!(buf3.as_slice(), &[1, 0, 1, 0]);
}

#[test]
fn gna_q_primitive_methods() {
    let scheme = GnaDevice::default_quant_scheme();
    let buf = GnaBuffer::from_bytes(vec![1, 2, 3, 4]);
    let prim = GnaQTensorPrimitive::new(Shape::from([2, 2]), scheme, buf);

    assert_eq!(prim.scheme(), scheme);
    assert_eq!(prim.dtype(), DType::QFloat(scheme));
    assert_eq!(prim.shape(), Shape::from([2, 2]));
    assert_eq!(prim.device(), GnaDevice::default());
    assert!(prim.can_mut());
    assert_eq!(prim.buffer().as_slice(), &[1, 2, 3, 4]);

    let reshaped = prim.with_shape(Shape::from([4]));
    assert_eq!(reshaped.shape(), Shape::from([4]));

    let (shape, s2, buf2) = reshaped.clone().into_parts();
    assert_eq!(shape, Shape::from([4]));
    assert_eq!(s2, scheme);
    assert_eq!(buf2.as_slice(), &[1, 2, 3, 4]);

    let buf3 = reshaped.into_buffer();
    assert_eq!(buf3.as_slice(), &[1, 2, 3, 4]);
}

// ---------------------------------------------------------------------------
// BoolTensorOps tests
// ---------------------------------------------------------------------------

#[test]
fn bool_creation_and_conversion() {
    let device = GnaDevice::default();

    let empty = GnaBackend::bool_empty(Shape::from([2, 2]), &device, BoolStore::Native);
    assert_eq!(read_bool(empty), vec![false, false, false, false]);

    let zeros = GnaBackend::bool_zeros(Shape::from([3]), &device, BoolStore::Native);
    assert_eq!(read_bool(zeros), vec![false, false, false]);

    let ones = GnaBackend::bool_ones(Shape::from([2, 2]), &device, BoolStore::Native);
    assert_eq!(read_bool(ones), vec![true, true, true, true]);

    let t = bool_tensor(&[true, false, true], &[3]);
    let t_dev = GnaBackend::bool_to_device(t.clone(), &device);
    assert_eq!(read_bool(t_dev), vec![true, false, true]);

    let as_int = GnaBackend::bool_into_int(t.clone(), IntDType::I32);
    assert_eq!(read_i32(as_int), vec![1, 0, 1]);

    let as_float = GnaBackend::bool_into_float(t, FloatDType::F32);
    assert_eq!(read_f32(as_float), vec![1.0, 0.0, 1.0]);
}

#[test]
fn bool_logical_and_comparison() {
    let a = bool_tensor(&[true, true, false, false], &[4]);
    let b = bool_tensor(&[true, false, true, false], &[4]);

    assert_eq!(
        read_bool(GnaBackend::bool_and(a.clone(), b.clone())),
        vec![true, false, false, false]
    );
    assert_eq!(
        read_bool(GnaBackend::bool_or(a.clone(), b.clone())),
        vec![true, true, true, false]
    );
    assert_eq!(
        read_bool(GnaBackend::bool_not(a.clone())),
        vec![false, false, true, true]
    );
    assert_eq!(
        read_bool(GnaBackend::bool_equal(a.clone(), b)),
        vec![true, false, false, true]
    );
    assert_eq!(
        read_bool(GnaBackend::bool_equal_elem(
            a,
            Scalar::new(true, &DType::Bool(BoolStore::Native))
        )),
        vec![true, true, false, false]
    );
}

#[test]
fn bool_shape_and_layout_ops() {
    let t = bool_tensor(&[true, false, false, true], &[2, 2]);

    let reshaped = GnaBackend::bool_reshape(t.clone(), Shape::from([4]));
    assert_eq!(reshaped.shape(), Shape::from([4]));
    assert_eq!(read_bool(reshaped), vec![true, false, false, true]);

    let swapped = GnaBackend::bool_swap_dims(t.clone(), 0, 1);
    assert_eq!(swapped.shape(), Shape::from([2, 2]));
    assert_eq!(read_bool(swapped), vec![true, false, false, true]);

    let t2 = bool_tensor(&[true, false, true, false, false, true], &[2, 3]);
    let permuted = GnaBackend::bool_permute(t2.clone(), &[1, 0]);
    assert_eq!(permuted.shape(), Shape::from([3, 2]));
    assert_eq!(
        read_bool(permuted),
        vec![true, false, false, false, true, true]
    );

    let flipped = GnaBackend::bool_flip(t2, &[1]);
    assert_eq!(
        read_bool(flipped),
        vec![true, false, true, true, false, false]
    );

    let row = bool_tensor(&[true, false], &[1, 2]);
    let expanded = GnaBackend::bool_expand(row, Shape::from([3, 2]));
    assert_eq!(expanded.shape(), Shape::from([3, 2]));
    assert_eq!(
        read_bool(expanded),
        vec![true, false, true, false, true, false]
    );

    let seq = bool_tensor(&[true, false, true, true], &[4]);
    let unfolded = GnaBackend::bool_unfold(seq, 0, 2, 1);
    assert_eq!(unfolded.shape(), Shape::from([3, 2]));
    assert_eq!(
        read_bool(unfolded),
        vec![true, false, false, true, true, true]
    );
}

#[test]
fn bool_slice_mask_and_indexing() {
    let t = bool_tensor(&[true, false, false, true, true, false], &[2, 3]);

    let sliced = GnaBackend::bool_slice(t.clone(), &[Slice::from(0..1), Slice::from(1..3)]);
    assert_eq!(sliced.shape(), Shape::from([1, 2]));
    assert_eq!(read_bool(sliced), vec![false, false]);

    let val = bool_tensor(&[true, true], &[1, 2]);
    let assigned =
        GnaBackend::bool_slice_assign(t.clone(), &[Slice::from(0..1), Slice::from(1..3)], val);
    assert_eq!(
        read_bool(assigned),
        vec![true, true, true, true, true, false]
    );

    let base = bool_tensor(&[false, false, false, false], &[4]);
    let mask = bool_tensor(&[true, false, true, false], &[4]);
    let src = bool_tensor(&[true, true, false, true], &[4]);
    assert_eq!(
        read_bool(GnaBackend::bool_mask_where(base.clone(), mask.clone(), src)),
        vec![true, false, false, false]
    );
    assert_eq!(
        read_bool(GnaBackend::bool_mask_fill(
            base,
            mask,
            Scalar::new(true, &DType::Bool(BoolStore::Native))
        )),
        vec![true, false, true, false]
    );

    let g_src = bool_tensor(&[true, false, true, false], &[2, 2]);
    let g_idx = i32_tensor(&[1, 0, 0, 1], &[2, 2]);
    let gathered = GnaBackend::bool_gather(1, g_src.clone(), g_idx.clone());
    assert_eq!(read_bool(gathered), vec![false, true, true, false]);

    let s_base = bool_tensor(&[false, false, false, false], &[2, 2]);
    let s_val = bool_tensor(&[true, false, false, true], &[2, 2]);
    let scattered = GnaBackend::bool_scatter_or(1, s_base, g_idx, s_val);
    assert_eq!(read_bool(scattered), vec![false, true, false, true]);

    let sel_idx = i32_tensor(&[1], &[1]);
    let selected = GnaBackend::bool_select(g_src.clone(), 0, sel_idx.clone());
    assert_eq!(selected.shape(), Shape::from([1, 2]));
    assert_eq!(read_bool(selected), vec![true, false]);

    let sel_base = bool_tensor(&[false, false, false, false], &[2, 2]);
    let sel_val = bool_tensor(&[true, true], &[1, 2]);
    let sel_or = GnaBackend::bool_select_or(sel_base, 0, sel_idx, sel_val);
    assert_eq!(read_bool(sel_or), vec![false, false, true, true]);
}

// ---------------------------------------------------------------------------
// QTensorOps tests
// ---------------------------------------------------------------------------

#[test]
fn q_quantize_dequantize_and_data() {
    let device = GnaDevice::default();
    let scheme = GnaDevice::default_quant_scheme();

    let input = GnaBackend::float_from_data(TensorData::from([0.0f32, 1.0, -2.0, 3.0]), &device);
    let scale = GnaBackend::float_from_data(TensorData::from([0.5f32]), &device);
    let qparams = QuantizationParametersPrimitive {
        scales: scale,
        global: None,
    };

    let q = GnaBackend::quantize(input, &scheme, qparams);
    assert_eq!(q.shape(), Shape::from([4]));
    assert_eq!(q.scheme(), scheme);

    let q_dev = GnaBackend::q_to_device(q.clone(), &device);
    assert_eq!(q_dev.shape(), Shape::from([4]));

    let q_data = try_read_sync(GnaBackend::q_into_data(q.clone()))
        .expect("sync read")
        .expect("q_into_data");
    let q_roundtrip = GnaBackend::q_from_data(q_data, &device);
    assert_eq!(q_roundtrip.shape(), Shape::from([4]));

    let deq = GnaBackend::dequantize(q, FloatDType::F32);
    assert_eq!(read_f32(deq), vec![0.0, 1.0, -2.0, 3.0]);
}

#[test]
fn q_layout_ops() {
    let device = GnaDevice::default();
    let scheme = GnaDevice::default_quant_scheme();
    let scale = GnaBackend::float_from_data(TensorData::from([1.0f32]), &device);
    let qparams = QuantizationParametersPrimitive {
        scales: scale.clone(),
        global: None,
    };
    let f_base = GnaBackend::float_from_data(
        TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        &device,
    );
    let q_base = GnaBackend::quantize(f_base, &scheme, qparams);

    let reshaped = GnaBackend::q_reshape(q_base.clone(), Shape::from([3, 2]));
    assert_eq!(reshaped.shape(), Shape::from([3, 2]));

    let swapped = GnaBackend::q_swap_dims(q_base.clone(), 0, 1);
    assert_eq!(swapped.shape(), Shape::from([3, 2]));
    assert_eq!(
        read_f32(GnaBackend::dequantize(swapped, FloatDType::F32)),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    let permuted = GnaBackend::q_permute(q_base.clone(), &[1, 0]);
    assert_eq!(permuted.shape(), Shape::from([3, 2]));
    assert_eq!(
        read_f32(GnaBackend::dequantize(permuted, FloatDType::F32)),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    let flipped = GnaBackend::q_flip(q_base.clone(), &[1]);
    assert_eq!(
        read_f32(GnaBackend::dequantize(flipped, FloatDType::F32)),
        vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]
    );

    let f_row = GnaBackend::float_from_data(TensorData::from([[10.0f32, 20.0]]), &device);
    let row = GnaBackend::quantize(
        f_row,
        &scheme,
        QuantizationParametersPrimitive {
            scales: scale,
            global: None,
        },
    );
    let expanded = GnaBackend::q_expand(row, Shape::from([2, 2]));
    assert_eq!(expanded.shape(), Shape::from([2, 2]));
    assert_eq!(
        read_f32(GnaBackend::dequantize(expanded, FloatDType::F32)),
        vec![10.0, 20.0, 10.0, 20.0]
    );

    let idx = GnaBackend::int_from_data(TensorData::from([1i32]), &device);
    let selected = GnaBackend::q_select(q_base.clone(), 0, idx);
    assert_eq!(selected.shape(), Shape::from([1, 3]));
    assert_eq!(
        read_f32(GnaBackend::dequantize(selected, FloatDType::F32)),
        vec![4.0, 5.0, 6.0]
    );

    let sliced = GnaBackend::q_slice(q_base, &[Slice::from(0..2), Slice::from(1..3)]);
    assert_eq!(sliced.shape(), Shape::from([2, 2]));
    assert_eq!(
        read_f32(GnaBackend::dequantize(sliced, FloatDType::F32)),
        vec![2.0, 3.0, 5.0, 6.0]
    );
}
