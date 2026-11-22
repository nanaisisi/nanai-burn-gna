use crate::device::GnaDevice;
use crate::tensor::{
    GnaBoolTensorPrimitive, GnaFloatTensorPrimitive, GnaIntTensorPrimitive, GnaQTensorPrimitive,
};
use burn_backend::distributed::DistributedOps;
use burn_backend::ops::{
    ActivationOps, TransactionOps, TransactionPrimitive, TransactionPrimitiveData,
};
use burn_backend::{
    Backend, BackendTypes, DType, DTypeUsage, DTypeUsageSet, ExecutionError, GraphUnsupported,
};
use enumset::{EnumSet, enum_set};

mod bool;
mod float;
mod int;
mod module;
mod quantized;

/// Intel GNA Backend for Burn deep learning framework.
#[derive(Clone, Copy, Default, Debug)]
pub struct GnaBackend;

impl BackendTypes for GnaBackend {
    type Device = GnaDevice;
    type FloatTensorPrimitive = GnaFloatTensorPrimitive;
    type IntTensorPrimitive = GnaIntTensorPrimitive;
    type BoolTensorPrimitive = GnaBoolTensorPrimitive;
    type QuantizedTensorPrimitive = GnaQTensorPrimitive;
    type GraphPrimitive = GraphUnsupported;
}

impl Backend for GnaBackend {
    fn name(device: &Self::Device) -> String {
        format!("gna (device #{})", device.index)
    }

    fn seed(_device: &Self::Device, _seed: u64) {
        // GNA accelerator is deterministic inference device; seed is no-op
    }

    fn dtype_usage(_device: &Self::Device, dtype: DType) -> DTypeUsageSet {
        match dtype {
            DType::QFloat(_) => enum_set!(DTypeUsage::Storage | DTypeUsage::Accelerated),
            DType::Bool(_) | DType::F32 | DType::I32 | DType::I16 | DType::I8 => {
                enum_set!(DTypeUsage::Storage | DTypeUsage::Arithmetic | DTypeUsage::Accelerated)
            }
            _ => EnumSet::empty(),
        }
    }

    fn device_count(_type_id: u16) -> usize {
        // Check dynamically if GNA library is present, fallback to 0/1
        if let Ok(lib) = nanai_gna_dll_load::GnaLibrary::load_default() {
            nanai_gna_dll_load::GnaDevice::get_count(&lib).unwrap_or(0) as usize
        } else {
            0
        }
    }

    fn sync(_device: &Self::Device) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn flush(_device: &Self::Device) -> Result<(), ExecutionError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DistributedOps
// ---------------------------------------------------------------------------
impl DistributedOps<Self> for GnaBackend {}

// ---------------------------------------------------------------------------
// ActivationOps
// ---------------------------------------------------------------------------
impl ActivationOps<Self> for GnaBackend {}

// ---------------------------------------------------------------------------
// TransactionOps
// ---------------------------------------------------------------------------
impl TransactionOps<Self> for GnaBackend {
    async fn tr_execute(
        _transaction: TransactionPrimitive<Self>,
    ) -> Result<TransactionPrimitiveData, ExecutionError> {
        Err(ExecutionError::WithContext {
            reason: "TransactionOps not supported on GNA backend".into(),
        })
    }
}
