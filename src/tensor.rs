use crate::device::GnaDevice;
use burn_backend::quantization::QuantScheme;
use burn_backend::{BoolStore, DType, FloatDType, IntDType, Shape, TensorMetadata};
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

/// Float tensor primitive for GnaBackend.
#[derive(Clone, Debug)]
pub struct GnaFloatTensorPrimitive {
    pub shape: Shape,
    pub dtype: FloatDType,
    pub buffer: GnaBuffer,
}

impl TensorMetadata for GnaFloatTensorPrimitive {
    type Device = GnaDevice;

    fn dtype(&self) -> DType {
        self.dtype.into()
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn device(&self) -> Self::Device {
        GnaDevice::default()
    }

    fn can_mut(&self) -> bool {
        self.buffer.can_mut()
    }
}

/// Int tensor primitive for GnaBackend.
#[derive(Clone, Debug)]
pub struct GnaIntTensorPrimitive {
    pub shape: Shape,
    pub dtype: IntDType,
    pub buffer: GnaBuffer,
}

impl TensorMetadata for GnaIntTensorPrimitive {
    type Device = GnaDevice;

    fn dtype(&self) -> DType {
        self.dtype.into()
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn device(&self) -> Self::Device {
        GnaDevice::default()
    }

    fn can_mut(&self) -> bool {
        self.buffer.can_mut()
    }
}

/// Bool tensor primitive for GnaBackend.
#[derive(Clone, Debug)]
pub struct GnaBoolTensorPrimitive {
    pub shape: Shape,
    pub buffer: GnaBuffer,
}

impl TensorMetadata for GnaBoolTensorPrimitive {
    type Device = GnaDevice;

    fn dtype(&self) -> DType {
        DType::Bool(BoolStore::Native)
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn device(&self) -> Self::Device {
        GnaDevice::default()
    }

    fn can_mut(&self) -> bool {
        self.buffer.can_mut()
    }
}

/// Quantized tensor primitive for GnaBackend.
#[derive(Clone, Debug)]
pub struct GnaQTensorPrimitive {
    pub shape: Shape,
    pub scheme: QuantScheme,
    pub buffer: GnaBuffer,
}

impl TensorMetadata for GnaQTensorPrimitive {
    type Device = GnaDevice;

    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn device(&self) -> Self::Device {
        GnaDevice::default()
    }

    fn can_mut(&self) -> bool {
        self.buffer.can_mut()
    }
}

impl GnaQTensorPrimitive {
    pub fn default_scheme() -> QuantScheme {
        GnaDevice::default_quant_scheme()
    }
}
