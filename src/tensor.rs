use burn::tensor::backend::QTensorPrimitive;
use burn::tensor::quantization::{QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue};
use burn::tensor::{BoolStore, DType, FloatDType, IntDType, Shape, TensorMetadata};
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
    fn dtype(&self) -> DType {
        self.dtype.into()
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
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
    fn dtype(&self) -> DType {
        self.dtype.into()
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }
}

/// Bool tensor primitive for GnaBackend.
#[derive(Clone, Debug)]
pub struct GnaBoolTensorPrimitive {
    pub shape: Shape,
    pub buffer: GnaBuffer,
}

impl TensorMetadata for GnaBoolTensorPrimitive {
    fn dtype(&self) -> DType {
        DType::Bool(BoolStore::Native)
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
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
    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }
}

impl QTensorPrimitive for GnaQTensorPrimitive {
    fn scheme(&self) -> &QuantScheme {
        &self.scheme
    }

    fn default_scheme() -> QuantScheme {
        // Intel GNA commonly operates with INT8 quantized representations
        QuantScheme::default()
            .with_value(QuantValue::Q8F)
            .with_mode(QuantMode::Symmetric)
            .with_param(QuantParam::F32)
            .with_store(QuantStore::Native)
    }
}
