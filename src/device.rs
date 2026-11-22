use burn_backend::quantization::{QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype};
use burn_backend::{BoolStore, DType, Device, DeviceId, DeviceOps, DeviceSettings};
use burn_std::QuantConfig;

/// Intel GNA device representation for Burn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct GnaDevice {
    /// Device index identifier (typically 0 for the first GNA device)
    index: u16,
}

impl GnaDevice {
    /// Create a new GNA device with the specified index.
    pub const fn new(index: u16) -> Self {
        Self { index }
    }

    /// Return the device index identifier.
    pub const fn index(&self) -> u16 {
        self.index
    }

    /// Get the count of available GNA hardware devices.
    pub fn get_count(lib: &crate::GnaLibrary) -> crate::Result<u32> {
        Ok(crate::GnaHwDevice::get_count(lib)?)
    }

    /// Query the hardware version of a GNA device.
    pub fn get_version(
        lib: &crate::GnaLibrary,
        device_index: u32,
    ) -> crate::Result<crate::Gna2DeviceVersion> {
        Ok(crate::GnaHwDevice::get_version(lib, device_index)?)
    }

    /// Open a GNA device session.
    pub fn open(lib: &crate::GnaLibrary, device_index: u32) -> crate::Result<crate::GnaHwDevice> {
        Ok(crate::GnaHwDevice::open(lib, device_index)?)
    }

    /// Default quantization scheme for Intel GNA (symmetric INT8 with F32 scale).
    pub fn default_quant_scheme() -> QuantScheme {
        QuantScheme::default()
            .with_value(QuantValue::Q8F)
            .with_mode(QuantMode::Symmetric)
            .per_tensor(ScaleDtype::F32)
            .with_store(QuantStore::Native)
    }
}

impl Device for GnaDevice {
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
    fn defaults(&self) -> DeviceSettings {
        DeviceSettings::new(
            DType::F32,
            DType::I32,
            DType::Bool(BoolStore::Native),
            QuantConfig::new(Self::default_quant_scheme(), Default::default()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::Device;

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
