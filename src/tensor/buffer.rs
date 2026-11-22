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

    /// Return the buffer contents as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Bytes(bytes) => bytes.as_slice(),
        }
    }

    /// Return the byte length of the buffer.
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Return `true` if the buffer has zero bytes.
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    /// Copy the buffer contents into a `Vec<u8>`.
    pub fn to_vec(&self) -> Vec<u8> {
        self.as_slice().to_vec()
    }

    /// Consume the buffer and return its contents as a `Vec<u8>`, avoiding a copy when uniquely owned.
    pub fn into_vec(self) -> Vec<u8> {
        match self {
            Self::Bytes(bytes) => Arc::unwrap_or_clone(bytes),
        }
    }

    /// Check if this buffer is uniquely owned and can be mutated in-place.
    pub fn can_mut(&self) -> bool {
        match self {
            Self::Bytes(arc) => Arc::strong_count(arc) == 1,
        }
    }
}
