use std::sync::Arc;

/// Underlying buffer representation for GNA tensor primitives.
#[derive(Clone, Debug)]
pub enum GnaBuffer {
    /// Raw byte buffer (e.g. host-allocated or staging buffer)
    Bytes(Arc<Vec<u8>>),
}

impl GnaBuffer {
    /// Create a buffer from a byte vector.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self::Bytes(Arc::new(bytes))
    }

    /// Create a zero-initialized buffer of the given byte length.
    pub fn zeros(len: usize) -> Self {
        Self::from_bytes(vec![0u8; len])
    }

    /// Create a buffer filled with a repeated byte value.
    pub fn filled(value: u8, len: usize) -> Self {
        Self::from_bytes(vec![value; len])
    }

    /// Copy the buffer contents into a `Vec<u8>`.
    pub fn to_vec(&self) -> Vec<u8> {
        match self {
            Self::Bytes(bytes) => bytes.as_slice().to_vec(),
        }
    }

    /// Check if this buffer is uniquely owned and can be mutated in-place.
    pub fn can_mut(&self) -> bool {
        match self {
            Self::Bytes(arc) => Arc::strong_count(arc) == 1,
        }
    }
}
