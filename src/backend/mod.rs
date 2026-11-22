use crate::device::GnaDevice;
use crate::tensor::{
    GnaBoolTensorPrimitive, GnaFloatTensorPrimitive, GnaIntTensorPrimitive, GnaQTensorPrimitive,
};
use burn_backend::distributed::DistributedOps;
use burn_backend::ops::{ActivationOps, TransactionOps};
use burn_backend::{
    Backend, BackendTypes, DType, DTypeUsage, DTypeUsageSet, ExecutionError, GraphUnsupported,
};
use enumset::{EnumSet, enum_set};

mod bool;
mod float;
pub(crate) mod gna_exec;
mod int;
mod module;
mod quantized;

pub(crate) struct UnfoldParams {
    pub dim: usize,
    pub size: usize,
    pub step: usize,
}

impl UnfoldParams {
    pub(crate) const fn new(dim: usize, size: usize, step: usize) -> Self {
        Self { dim, size, step }
    }
}

/// Intel GNA Backend for Burn deep learning framework.
#[derive(Clone, Copy, Default, Debug)]
pub struct GnaBackend;

impl GnaBackend {
    /// Return the hardware performance statistics from the most recent GNA execution.
    pub fn last_performance_stats() -> Option<crate::ffi::GnaPerformanceStats> {
        gna_exec::last_performance_stats()
    }

    /// Return the total bytes currently allocated via GNA hardware memory allocators.
    pub fn allocated_bytes() -> usize {
        crate::ffi::gna_allocated_bytes()
    }

    /// Check whether the GNA device is currently uncontended by other processes.
    pub fn is_uncontended() -> crate::Result<bool> {
        Ok(crate::ffi::is_gna_uncontended()?)
    }
}

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
        format!("gna (device #{})", device.index())
    }

    fn seed(_device: &Self::Device, seed: u64) {
        gna_exec::extra::set_rng_seed(seed);
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
        if let Ok(lib) = crate::GnaLibrary::load_default() {
            GnaDevice::get_count(&lib).unwrap_or(0) as usize
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
impl TransactionOps<Self> for GnaBackend {}
