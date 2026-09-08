pub mod backend;
pub mod device;
pub mod tensor;

pub use backend::GnaBackend;
pub use device::GnaDevice;
pub use nanai_gna_dll_rs::*;
pub use tensor::{
    GnaBoolTensorPrimitive, GnaBuffer, GnaFloatTensorPrimitive, GnaIntTensorPrimitive,
    GnaQTensorPrimitive,
};

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend;
    use burn::tensor::Tensor;

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
        let tensor = Tensor::<GnaBackend, 2>::from_data([[1.0f32, 2.0], [3.0, 4.0]], &device);
        let shape = tensor.shape();
        assert_eq!(shape.dims(), [2, 2]);

        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().expect("slice f32");
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn gna_int_tensor_creation() {
        let device = GnaDevice::default();
        let tensor = Tensor::<GnaBackend, 1, burn::tensor::Int>::from_data([10i32, 20, 30], &device);
        let shape = tensor.shape();
        assert_eq!(shape.dims(), [3]);

        let data = tensor.into_data();
        let slice = data.as_slice::<i32>().expect("slice i32");
        assert_eq!(slice, &[10, 20, 30]);
    }
}

