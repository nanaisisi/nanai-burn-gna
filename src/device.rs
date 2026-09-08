use burn::tensor::backend::{DeviceId, DeviceOps};

/// Intel GNA device representation for Burn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[derive(Default)]
pub struct GnaDevice {
    /// Device index identifier (typically 0 for the first GNA device)
    pub index: u16,
}

impl GnaDevice {
    /// Create a new GNA device with the specified index.
    pub const fn new(index: u16) -> Self {
        Self { index }
    }

    /// Get the count of available GNA hardware devices.
    pub fn get_count(lib: &nanai_gna_dll_rs::GnaLibrary) -> nanai_gna_dll_rs::Result<u32> {
        nanai_gna_dll_rs::GnaDevice::get_count(lib)
    }

    /// Query the hardware version of a GNA device.
    pub fn get_version(
        lib: &nanai_gna_dll_rs::GnaLibrary,
        device_index: u32,
    ) -> nanai_gna_dll_rs::Result<nanai_gna_dll_rs::Gna2DeviceVersion> {
        nanai_gna_dll_rs::GnaDevice::get_version(lib, device_index)
    }

    /// Open a GNA device session.
    pub fn open(
        lib: &nanai_gna_dll_rs::GnaLibrary,
        device_index: u32,
    ) -> nanai_gna_dll_rs::Result<nanai_gna_dll_rs::GnaDevice> {
        nanai_gna_dll_rs::GnaDevice::open(lib, device_index)
    }
}


impl burn::tensor::backend::Device for GnaDevice {
    fn from_id(device_id: DeviceId) -> Self {
        Self {
            index: device_id.index_id,
        }
    }

    fn to_id(&self) -> DeviceId {
        // Use a unique type_id for GNA device, e.g. 0x474e ('G', 'N')
        DeviceId::new(0x474e, self.index)
    }
}

impl DeviceOps for GnaDevice {
    fn id(&self) -> DeviceId {
        burn::tensor::backend::Device::to_id(self)
    }

    fn inner(&self) -> &Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Device;

    #[test]
    fn test_device_id_conversion() {
        let dev = GnaDevice { index: 1 };
        let id = dev.to_id();
        assert_eq!(id.type_id, 0x474e);
        assert_eq!(id.index_id, 1);

        let restored = GnaDevice::from_id(id);
        assert_eq!(dev, restored);
    }
}
