use burn_backend::backend::ExecutionError;
use burn_backend::ops::{
    AttentionModuleOptions, BoolTensorOps, ConvOptions, ConvTransposeOptions, DeformConvOptions,
    FloatTensorOps, IntTensorOps, InterpolateMode, InterpolateOptions, ModuleOps, QTensorOps,
    TransactionOps, TransactionPrimitive,
};
use burn_backend::tensor::{BoolTensor, FloatTensor};
use burn_backend::{
    Backend, BoolDType, BoolStore, DType, DTypeUsage, Device, DeviceId, DeviceOps, FloatDType,
    TensorData,
};
use nanai_burn_gna::{GnaBackend, GnaBackendError, GnaDevice};

fn dev() -> GnaDevice {
    GnaDevice::new(0)
}

fn f32_tensor(shape: &[usize], data: Vec<f32>) -> FloatTensor<GnaBackend> {
    let td = TensorData::new(data, shape.to_vec());
    GnaBackend::float_from_data(td, &dev())
}

fn bool_tensor(shape: &[usize], data: Vec<bool>) -> BoolTensor<GnaBackend> {
    let td = TensorData::new(data, shape.to_vec());
    GnaBackend::bool_from_data(td, &dev())
}

fn poll_ready<F: std::future::Future>(fut: F) -> F::Output {
    let waker = std::task::Waker::noop();
    let mut cx = std::task::Context::from_waker(waker);
    let mut pinned = std::pin::pin!(fut);
    match std::future::Future::poll(pinned.as_mut(), &mut cx) {
        std::task::Poll::Ready(out) => out,
        std::task::Poll::Pending => panic!("future did not resolve immediately"),
    }
}

fn to_f32_vec(t: FloatTensor<GnaBackend>) -> Vec<f32> {
    poll_ready(GnaBackend::float_into_data(t))
        .expect("float_into_data failed")
        .try_to_vec::<f32>()
        .expect("expected f32 tensor data")
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "index {i}: actual {a} != expected {e} (tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// ModuleOps: conv3d, conv_transpose2d, conv_transpose3d
// ---------------------------------------------------------------------------

#[test]
fn test_conv3d_basic_and_bias() {
    // x: [1, 1, 2, 2, 2], weight: [1, 1, 2, 2, 2]
    let x = f32_tensor(
        &[1, 1, 2, 2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    let w = f32_tensor(&[1, 1, 2, 2, 2], vec![1.0; 8]);
    let b = f32_tensor(&[1], vec![0.5]);

    let opts = ConvOptions::new([1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
    let out_no_bias = GnaBackend::conv3d(x.clone(), w.clone(), None, opts.clone());
    assert_close(&to_f32_vec(out_no_bias), &[36.0], 1e-5);

    let out_bias = GnaBackend::conv3d(x, w, Some(b), opts);
    assert_close(&to_f32_vec(out_bias), &[36.5], 1e-5);
}

#[test]
fn test_conv_transpose2d_basic() {
    // x: [1, 1, 2, 2], weight: [1, 1, 2, 2]
    let x = f32_tensor(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let w = f32_tensor(&[1, 1, 2, 2], vec![1.0; 4]);
    let b = f32_tensor(&[1], vec![1.0]);

    let opts = ConvTransposeOptions::new([1, 1], [0, 0], [0, 0], [1, 1], 1);
    let out = GnaBackend::conv_transpose2d(x, w, Some(b), opts);
    // Output shape [1, 1, 3, 3]:
    // [1, 1+2, 2; 1+3, 1+2+3+4, 2+4; 3, 3+4, 4] + 1.0
    let expected = vec![2.0, 4.0, 3.0, 5.0, 11.0, 7.0, 4.0, 8.0, 5.0];
    assert_close(&to_f32_vec(out), &expected, 1e-5);
}

#[test]
fn test_conv_transpose3d_basic() {
    // x: [1, 1, 1, 2, 2], weight: [1, 1, 2, 2, 2]
    let x = f32_tensor(&[1, 1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let w = f32_tensor(&[1, 1, 2, 2, 2], vec![1.0; 8]);
    let opts = ConvTransposeOptions::new([1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], 1);
    let out = GnaBackend::conv_transpose3d(x, w, None, opts);
    let data = to_f32_vec(out);
    // Output shape [1, 1, 2, 3, 3] -> 18 elements
    assert_eq!(data.len(), 18);
    // Both depth slices are identical: [1, 3, 2, 4, 10, 6, 3, 7, 4]
    let slice = [1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0];
    assert_close(&data[0..9], &slice, 1e-5);
    assert_close(&data[9..18], &slice, 1e-5);
}

// ---------------------------------------------------------------------------
// ModuleOps: deform_conv2d & deform_conv2d_backward
// ---------------------------------------------------------------------------

#[test]
fn test_deform_conv2d_and_backward() {
    // x: [1, 1, 2, 2], weight: [1, 1, 2, 2], zero offset -> matches standard conv2d
    let x = f32_tensor(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let offset = f32_tensor(&[1, 8, 1, 1], vec![0.0; 8]);
    let weight = f32_tensor(&[1, 1, 2, 2], vec![1.0, 1.0, 1.0, 1.0]);
    let mask = f32_tensor(&[1, 4, 1, 1], vec![1.0; 4]);
    let bias = f32_tensor(&[1], vec![2.0]);
    let opts = DeformConvOptions::new([1, 1], [0, 0], [1, 1], 1, 1);

    let out = GnaBackend::deform_conv2d(
        x.clone(),
        offset.clone(),
        weight.clone(),
        Some(mask.clone()),
        Some(bias.clone()),
        opts.clone(),
    );
    assert_close(&to_f32_vec(out), &[12.0], 1e-5);

    let grad_out = f32_tensor(&[1, 1, 1, 1], vec![1.0]);
    let bwd = GnaBackend::deform_conv2d_backward(
        x,
        offset,
        weight,
        Some(mask),
        Some(bias),
        grad_out,
        opts,
    );
    assert_close(&to_f32_vec(bwd.x_grad), &[1.0, 1.0, 1.0, 1.0], 1e-5);
    assert_close(&to_f32_vec(bwd.weight_grad), &[1.0, 2.0, 3.0, 4.0], 1e-5);
    assert!(bwd.bias_grad.is_some());
    assert_close(&to_f32_vec(bwd.bias_grad.unwrap()), &[1.0], 1e-5);
    assert!(bwd.mask_grad.is_some());
}

// ---------------------------------------------------------------------------
// ModuleOps: adaptive_avg_pool2d / 3d and backward
// ---------------------------------------------------------------------------

#[test]
fn test_adaptive_avg_pool2d_and_backward() {
    let x = f32_tensor(&[1, 1, 2, 2], vec![1.0, 3.0, 5.0, 7.0]);
    let out = GnaBackend::adaptive_avg_pool2d(x.clone(), [1, 1]);
    assert_close(&to_f32_vec(out), &[4.0], 1e-5);

    let grad = f32_tensor(&[1, 1, 1, 1], vec![4.0]);
    let x_grad = GnaBackend::adaptive_avg_pool2d_backward(x, grad);
    assert_close(&to_f32_vec(x_grad), &[1.0, 1.0, 1.0, 1.0], 1e-5);
}

#[test]
fn test_adaptive_avg_pool3d_and_backward() {
    let x = f32_tensor(
        &[1, 1, 2, 2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    let out = GnaBackend::adaptive_avg_pool3d(x.clone(), [1, 1, 1]);
    assert_close(&to_f32_vec(out), &[4.5], 1e-5);

    let grad = f32_tensor(&[1, 1, 1, 1, 1], vec![8.0]);
    let x_grad = GnaBackend::adaptive_avg_pool3d_backward(x, grad);
    assert_close(&to_f32_vec(x_grad), &[1.0; 8], 1e-5);
}

// ---------------------------------------------------------------------------
// ModuleOps: interpolate & interpolate_backward
// ---------------------------------------------------------------------------

#[test]
fn test_interpolate_and_backward_modes() {
    let x = f32_tensor(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    for mode in [
        InterpolateMode::Nearest,
        InterpolateMode::Bilinear,
        InterpolateMode::Bicubic,
    ] {
        let opts = InterpolateOptions::new(mode);
        let out = GnaBackend::interpolate(x.clone(), [4, 4], opts.clone());
        let out_vec = to_f32_vec(out);
        assert_eq!(out_vec.len(), 16);

        let grad = f32_tensor(&[1, 1, 4, 4], vec![1.0; 16]);
        let x_grad = GnaBackend::interpolate_backward(x.clone(), grad, [4, 4], opts);
        let grad_vec = to_f32_vec(x_grad);
        assert_eq!(grad_vec.len(), 4);
        // Total gradient sum is preserved (16.0)
        let sum: f32 = grad_vec.iter().sum();
        assert!((sum - 16.0).abs() < 1e-4, "sum={sum}");
    }
}

// ---------------------------------------------------------------------------
// ModuleOps: attention (causal, mask, bias, softcap, GQA)
// ---------------------------------------------------------------------------

#[test]
fn test_attention_variants() {
    // [batch=1, heads=2, seq=2, dim=2]
    let q = f32_tensor(&[1, 2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    let k = q.clone();
    let v = f32_tensor(&[1, 2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    // 1. Causal attention
    let causal_opts = AttentionModuleOptions {
        is_causal: true,
        ..Default::default()
    };
    let out_causal =
        GnaBackend::attention(q.clone(), k.clone(), v.clone(), None, None, causal_opts);
    let causal_vec = to_f32_vec(out_causal);
    assert_eq!(causal_vec.len(), 8);
    // First token of head 0 can only attend to itself -> v[0,0,0] = [1.0, 2.0]
    assert_close(&causal_vec[0..2], &[1.0, 2.0], 1e-5);

    // 2. Mask + Bias + Softcap
    // Mask: true = masked out
    let mask = bool_tensor(
        &[1, 2, 2, 2],
        vec![false, true, false, false, false, true, false, false],
    );
    let bias = f32_tensor(&[1, 2, 2, 2], vec![0.0; 8]);
    let softcap_opts = AttentionModuleOptions {
        softcap: Some(5.0),
        ..Default::default()
    };
    let out_masked = GnaBackend::attention(q, k, v, Some(mask), Some(bias), softcap_opts);
    let masked_vec = to_f32_vec(out_masked);
    assert_eq!(masked_vec.len(), 8);
    assert_close(&masked_vec[0..2], &[1.0, 2.0], 1e-5);

    // 3. GQA: 4 query heads, 2 kv heads
    let q_gqa = f32_tensor(&[1, 4, 2, 2], vec![0.5; 16]);
    let k_gqa = f32_tensor(&[1, 2, 2, 2], vec![0.5; 8]);
    let v_gqa = f32_tensor(&[1, 2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let out_gqa = GnaBackend::attention(
        q_gqa,
        k_gqa,
        v_gqa,
        None,
        None,
        AttentionModuleOptions::default(),
    );
    let gqa_vec = to_f32_vec(out_gqa);
    assert_eq!(gqa_vec.len(), 16);
    // Uniform q/k -> equal weights -> average of v0 and v1 per kv head
    assert_close(&gqa_vec[0..2], &[2.0, 3.0], 1e-5);
    assert_close(&gqa_vec[8..10], &[6.0, 7.0], 1e-5);
}

// ---------------------------------------------------------------------------
// Backend: dtype_usage, device_count, sync, flush, tr_execute
// ---------------------------------------------------------------------------

#[test]
fn test_backend_misc_ops() {
    let d = dev();

    let f32_usage = GnaBackend::dtype_usage(&d, DType::F32);
    assert!(f32_usage.contains(DTypeUsage::Storage));
    assert!(f32_usage.contains(DTypeUsage::Arithmetic));
    assert!(f32_usage.contains(DTypeUsage::Accelerated));

    let i32_usage = GnaBackend::dtype_usage(&d, DType::I32);
    assert!(i32_usage.contains(DTypeUsage::Arithmetic));

    let bool_usage = GnaBackend::dtype_usage(&d, DType::Bool(BoolStore::Native));
    assert!(bool_usage.contains(DTypeUsage::Storage));

    let q_usage = GnaBackend::dtype_usage(&d, DType::QFloat(GnaDevice::default_quant_scheme()));
    assert!(q_usage.contains(DTypeUsage::Storage));
    assert!(q_usage.contains(DTypeUsage::Accelerated));
    assert!(!q_usage.contains(DTypeUsage::Arithmetic));

    let f64_usage = GnaBackend::dtype_usage(&d, DType::F64);
    assert!(f64_usage.is_empty());

    // device_count should not panic
    let _count = GnaBackend::device_count(GnaDevice::TYPE_ID);

    // sync & flush
    assert!(GnaBackend::sync(&d).is_ok());
    assert!(GnaBackend::flush(&d).is_ok());

    // tr_execute (TransactionOps)
    let t_f = f32_tensor(&[2], vec![1.5, 2.5]);
    let mut tx = TransactionPrimitive::default();
    tx.read_floats.push(t_f);
    let res = poll_ready(GnaBackend::tr_execute(tx)).expect("tr_execute failed");
    assert_eq!(res.read_floats.len(), 1);
    assert_close(
        res.read_floats[0].try_to_vec::<f32>().unwrap().as_slice(),
        &[1.5, 2.5],
        1e-6,
    );
}

// ---------------------------------------------------------------------------
// Device & Error
// ---------------------------------------------------------------------------

#[test]
fn test_device_and_error() {
    let d = GnaDevice::new(3);
    assert_eq!(d.index(), 3);
    let id = d.to_id();
    assert_eq!(id, DeviceId::new(GnaDevice::TYPE_ID, 3));
    assert_eq!(GnaDevice::from_id(id), d);

    let settings = d.defaults();
    assert_eq!(settings.float_dtype, FloatDType::F32);
    assert_eq!(settings.int_dtype, DType::I32.into());
    assert_eq!(settings.bool_dtype, BoolDType::Native);

    let e1 = GnaBackendError::UnsupportedDType(DType::F64);
    assert!(e1.to_string().contains("Unsupported data type"));
    let exec_err: ExecutionError = e1.into();
    assert!(format!("{exec_err:?}").contains("Unsupported data type"));

    let e2 = GnaBackendError::DeviceNotInitialized(2);
    assert!(e2.to_string().contains("Device #2"));

    let e3 = GnaBackendError::InvalidMatmulRank(1);
    assert!(e3.to_string().contains("1"));

    let e4 = GnaBackendError::IntValueOutOfI16Range(40000);
    assert!(e4.to_string().contains("40000"));
}

#[test]
fn test_dtype_usage_extended() {
    let d = dev();
    for dt in [DType::F16, DType::BF16, DType::F64] {
        assert!(GnaBackend::dtype_usage(&d, dt).is_empty());
    }
    for qv in [
        burn_backend::quantization::QuantValue::Q4F,
        burn_backend::quantization::QuantValue::Q2F,
    ] {
        let scheme = burn_backend::quantization::QuantScheme::default().with_value(qv);
        let usage = GnaBackend::dtype_usage(&d, DType::QFloat(scheme));
        assert!(usage.contains(DTypeUsage::Storage));
        assert!(usage.contains(DTypeUsage::Accelerated));
    }
}

#[test]
#[should_panic]
fn test_invalid_matmul_rank_panics() {
    let a = f32_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f32_tensor(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let _ = GnaBackend::float_matmul(a, b);
}

#[test]
fn test_conv_and_pool_with_padding() {
    // 1. conv2d with padding > 0
    let x2 = f32_tensor(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let w2 = f32_tensor(&[1, 1, 2, 2], vec![1.0; 4]);
    let c2_opts = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);
    let out_c2 = GnaBackend::conv2d(x2.clone(), w2, None, c2_opts);
    // [1, 1, 3, 3] -> corners see 1 element, edges see 2, center sees 4
    assert_close(
        &to_f32_vec(out_c2),
        &[1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0],
        1e-5,
    );

    // 2. avg_pool2d with padding > 0
    let out_p2 = GnaBackend::avg_pool2d(x2, [2, 2], [1, 1], [1, 1], false, false);
    assert_eq!(to_f32_vec(out_p2).len(), 9);

    // 3. conv3d with padding > 0
    let x3 = f32_tensor(&[1, 1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let w3 = f32_tensor(&[1, 1, 2, 2, 2], vec![1.0; 8]);
    let c3_opts = ConvOptions::new([1, 1, 1], [1, 1, 1], [1, 1, 1], 1);
    let out_c3 = GnaBackend::conv3d(x3, w3, None, c3_opts);
    assert_eq!(to_f32_vec(out_c3).len(), 18);
}

#[test]
fn test_tr_execute_all_four_tensor_types() {
    let d = dev();
    let t_f = f32_tensor(&[2], vec![1.0, 2.0]);
    let t_i = GnaBackend::int_from_data(TensorData::new(vec![10i32, 20], [2]), &d);
    let t_b = bool_tensor(&[2], vec![true, false]);
    let t_q = GnaBackend::q_from_data(
        TensorData::quantized(
            vec![1i8, 2],
            [2],
            GnaDevice::default_quant_scheme(),
            &[1.0f32],
            None,
        ),
        &d,
    );

    let mut tx = TransactionPrimitive::default();
    tx.read_floats.push(t_f);
    tx.read_ints.push(t_i);
    tx.read_bools.push(t_b);
    tx.read_qfloats.push(t_q);

    let res = poll_ready(GnaBackend::tr_execute(tx)).expect("tr_execute failed");
    assert_eq!(res.read_floats.len(), 1);
    assert_eq!(res.read_ints.len(), 1);
    assert_eq!(res.read_bools.len(), 1);
    assert_eq!(res.read_qfloats.len(), 1);
}
