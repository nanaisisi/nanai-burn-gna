pub mod backend;
pub mod device;
pub mod error;
pub mod tensor;

pub use backend::GnaBackend;
pub use device::GnaDevice;
pub use error::{GnaBackendError, Result};
pub use nanai_gna_dll_load as ffi;
pub use nanai_gna_dll_load::{
    Gna2DeviceVersion, GnaBuffer as GnaHwBuffer, GnaDevice as GnaHwDevice, GnaError as GnaFfiError,
    GnaLibrary,
};
pub use tensor::{
    GnaBoolTensorPrimitive, GnaBuffer, GnaFloatTensorPrimitive, GnaIntTensorPrimitive,
    GnaQTensorPrimitive,
};

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::ops::{ConvOptions, FloatTensorOps, IntTensorOps, ModuleOps};
    use burn_backend::{Backend, TensorData, TensorMetadata, try_read_sync};

    #[test]
    fn load_gna_library() {
        let result = GnaLibrary::load_default();
        if let Err(err) = result {
            eprintln!("GNA load default: {err}");
        }
    }

    #[test]
    fn gna_backend_device_info() {
        let device = GnaDevice::new(0);
        let name = GnaBackend::name(&device);
        assert_eq!(name, "gna (device #0)");
    }

    #[test]
    fn gna_tensor_creation() {
        let device = GnaDevice::default();
        let input = TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]);
        let tensor = GnaBackend::float_from_data(input, &device);
        let shape = tensor.shape();
        assert_eq!(shape.dims(), [2, 2]);

        let data = try_read_sync(GnaBackend::float_into_data(tensor))
            .expect("sync read")
            .expect("float_into_data");
        let slice = data.as_slice::<f32>().expect("slice f32");
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn gna_int_tensor_creation() {
        let device = GnaDevice::default();
        let input = TensorData::from([10i32, 20, 30]);
        let tensor = GnaBackend::int_from_data(input, &device);
        let shape = tensor.shape();
        assert_eq!(shape.dims(), [3]);

        let data = try_read_sync(GnaBackend::int_into_data(tensor))
            .expect("sync read")
            .expect("int_into_data");
        let slice = data.as_slice::<i32>().expect("slice i32");
        assert_eq!(slice, &[10, 20, 30]);
    }

    #[test]
    fn gna_int_matmul_execution() {
        let device = GnaDevice::default();
        let lhs = GnaBackend::int_from_data(TensorData::from([[1i32, 2, 3], [4, 5, 6]]), &device);
        let rhs = GnaBackend::int_from_data(TensorData::from([[7i32, 8], [9, 1], [2, 3]]), &device);

        let out = GnaBackend::int_matmul(lhs, rhs);
        assert_eq!(out.shape().dims(), [2, 2]);

        let data = try_read_sync(GnaBackend::int_into_data(out))
            .expect("sync read")
            .expect("int_into_data");
        assert_eq!(
            data.as_slice::<i32>().expect("slice i32"),
            &[31, 19, 85, 55]
        );
    }

    #[test]
    fn gna_float_matmul_execution() {
        let device = GnaDevice::default();
        let lhs = GnaBackend::float_from_data(
            TensorData::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        );
        let rhs = GnaBackend::float_from_data(
            TensorData::from([[7.0f32, 8.0], [9.0, 1.0], [2.0, 3.0]]),
            &device,
        );

        let out = GnaBackend::float_matmul(lhs, rhs);
        assert_eq!(out.shape().dims(), [2, 2]);

        let data = try_read_sync(GnaBackend::float_into_data(out))
            .expect("sync read")
            .expect("float_into_data");
        let slice = data.as_slice::<f32>().expect("slice f32");
        let expected = [31.0f32, 19.0, 85.0, 55.0];
        for (actual, exp) in slice.iter().zip(expected.iter()) {
            assert!((actual - exp).abs() < 0.1, "expected {exp}, got {actual}");
        }
    }

    #[test]
    fn gna_arithmetic_ops() {
        let device = GnaDevice::default();

        let a = GnaBackend::int_from_data(TensorData::from([10i32, 20, 30]), &device);
        let b = GnaBackend::int_from_data(TensorData::from([1i32, 2, 3]), &device);
        let add_i = GnaBackend::int_add(a.clone(), b.clone());
        let sub_i = GnaBackend::int_sub(a.clone(), b.clone());
        let mul_i = GnaBackend::int_mul(a.clone(), b);
        let add_s = GnaBackend::int_add_scalar(a.clone(), 5i32.into());
        let mul_s = GnaBackend::int_mul_scalar(a, 2i32.into());

        assert_eq!(
            try_read_sync(GnaBackend::int_into_data(add_i))
                .unwrap()
                .unwrap()
                .as_slice::<i32>()
                .unwrap(),
            &[11, 22, 33]
        );
        assert_eq!(
            try_read_sync(GnaBackend::int_into_data(sub_i))
                .unwrap()
                .unwrap()
                .as_slice::<i32>()
                .unwrap(),
            &[9, 18, 27]
        );
        assert_eq!(
            try_read_sync(GnaBackend::int_into_data(mul_i))
                .unwrap()
                .unwrap()
                .as_slice::<i32>()
                .unwrap(),
            &[10, 40, 90]
        );
        assert_eq!(
            try_read_sync(GnaBackend::int_into_data(add_s))
                .unwrap()
                .unwrap()
                .as_slice::<i32>()
                .unwrap(),
            &[15, 25, 35]
        );
        assert_eq!(
            try_read_sync(GnaBackend::int_into_data(mul_s))
                .unwrap()
                .unwrap()
                .as_slice::<i32>()
                .unwrap(),
            &[20, 40, 60]
        );

        let fa = GnaBackend::float_from_data(TensorData::from([2.0f32, 4.0, 6.0]), &device);
        let fb = GnaBackend::float_from_data(TensorData::from([1.0f32, 2.0, 3.0]), &device);
        let f_add = GnaBackend::float_add(fa.clone(), fb.clone());
        let f_mul = GnaBackend::float_mul(fa, fb);
        let f_add_s = try_read_sync(GnaBackend::float_into_data(f_add))
            .unwrap()
            .unwrap();
        let f_mul_s = try_read_sync(GnaBackend::float_into_data(f_mul))
            .unwrap()
            .unwrap();
        for (a, e) in f_add_s
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .zip([3.0, 6.0, 9.0])
        {
            assert!((a - e).abs() < 0.05);
        }
        for (a, e) in f_mul_s
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .zip([2.0, 8.0, 18.0])
        {
            assert!((a - e).abs() < 0.05);
        }
    }

    #[test]
    fn gna_reduce_ops() {
        let device = GnaDevice::default();
        let t_i = GnaBackend::int_from_data(TensorData::from([[1i32, 2, 3], [4, 5, 6]]), &device);
        let sum_i = GnaBackend::int_sum(t_i);
        assert_eq!(
            try_read_sync(GnaBackend::int_into_data(sum_i))
                .unwrap()
                .unwrap()
                .as_slice::<i32>()
                .unwrap(),
            &[21]
        );

        let t_f =
            GnaBackend::float_from_data(TensorData::from([[2.0f32, 4.0], [6.0, 8.0]]), &device);
        let mean_f = GnaBackend::float_mean_dim(t_f, 1);
        let mean_d = try_read_sync(GnaBackend::float_into_data(mean_f))
            .unwrap()
            .unwrap();
        for (a, e) in mean_d.as_slice::<f32>().unwrap().iter().zip([3.0, 7.0]) {
            assert!((a - e).abs() < 0.05);
        }
    }

    #[test]
    fn gna_conv2d_execution() {
        let device = GnaDevice::default();
        let x = GnaBackend::float_from_data(
            TensorData::from([[[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
            &device,
        );
        let w =
            GnaBackend::float_from_data(TensorData::from([[[[1.0f32, 0.0], [0.0, 1.0]]]]), &device);
        let b = GnaBackend::float_from_data(TensorData::from([1.0f32]), &device);
        let out = GnaBackend::conv2d(x, w, Some(b), ConvOptions::new([1, 1], [0, 0], [1, 1], 1));
        assert_eq!(out.shape().dims(), [1, 1, 2, 2]);
        let data = try_read_sync(GnaBackend::float_into_data(out))
            .unwrap()
            .unwrap();
        for (a, e) in data
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .zip([7.0, 9.0, 13.0, 15.0])
        {
            assert!((a - e).abs() < 0.1, "expected {e}, got {a}");
        }
    }
}
