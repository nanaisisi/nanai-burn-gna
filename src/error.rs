use burn_backend::{DType, ExecutionError};

/// Error type for `nanai-burn-gna`.
#[derive(Debug, thiserror::Error)]
pub enum GnaBackendError {
    #[error("GNA hardware/driver error: {0}")]
    Ffi(#[from] nanai_gna_dll_load::GnaError),

    #[error("Unsupported data type for GNA backend: {0:?}")]
    UnsupportedDType(DType),

    #[error("Device #{0} is not initialized")]
    DeviceNotInitialized(u16),

    #[error("Tensor rank must be at least 2 for GNA matrix multiplication, got {0}")]
    InvalidMatmulRank(usize),

    #[error("Integer value {0} exceeds GNA hardware Int16 input/weight range [-32768, 32767]")]
    IntValueOutOfI16Range(i64),
}

/// Result type alias for `nanai-burn-gna`.
pub type Result<T> = std::result::Result<T, GnaBackendError>;

impl From<GnaBackendError> for ExecutionError {
    fn from(err: GnaBackendError) -> Self {
        ExecutionError::Generic {
            reason: err.to_string(),
            backtrace: Default::default(),
        }
    }
}
