use crate::device::GnaDevice;
use burn_backend::quantization::{QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype};
use burn_backend::{BoolStore, DType, FloatDType, IntDType, Shape, TensorMetadata};
use std::sync::Arc;

/// Underlying buffer representation for GNA tensor primitives.
#[derive(Clone, Debug)]
pub enum GnaBuffer {
    /// Raw byte buffer (e.g. host-allocated or staging buffer)
    Bytes(Arc<Vec<u8>>),
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
        match &self.buffer {
            GnaBuffer::Bytes(arc) => Arc::strong_count(arc) == 1,
        }
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
        match &self.buffer {
            GnaBuffer::Bytes(arc) => Arc::strong_count(arc) == 1,
        }
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
        match &self.buffer {
            GnaBuffer::Bytes(arc) => Arc::strong_count(arc) == 1,
        }
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
        match &self.buffer {
            GnaBuffer::Bytes(arc) => Arc::strong_count(arc) == 1,
        }
    }
}

impl GnaQTensorPrimitive {
    pub fn default_scheme() -> QuantScheme {
        // Intel GNA commonly operates with INT8 quantized representations
        QuantScheme::default()
            .with_value(QuantValue::Q8F)
            .with_mode(QuantMode::Symmetric)
            .per_tensor(ScaleDtype::F32)
            .with_store(QuantStore::Native)
    }
}
