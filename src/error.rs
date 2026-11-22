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
