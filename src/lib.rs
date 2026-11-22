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
    use burn_backend::ops::{FloatTensorOps, IntTensorOps};
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
}
