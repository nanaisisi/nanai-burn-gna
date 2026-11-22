use crate::device::GnaDevice;
use crate::tensor::buffer::GnaBuffer;
use burn_backend::{BoolStore, DType, Shape, TensorMetadata};

/// Bool tensor primitive for GnaBackend.
#[derive(Clone, Debug)]
pub struct GnaBoolTensorPrimitive {
    shape: Shape,
    buffer: GnaBuffer,
}

impl GnaBoolTensorPrimitive {
    /// Create a new bool tensor primitive.
    pub fn new(shape: Shape, buffer: GnaBuffer) -> Self {
        Self { shape, buffer }
    }

    /// Return a reference to the underlying buffer.
    pub fn buffer(&self) -> &GnaBuffer {
        &self.buffer
    }

    /// Consume the primitive and return its underlying buffer.
    pub fn into_buffer(self) -> GnaBuffer {
        self.buffer
    }

    /// Consume the primitive and return `(shape, buffer)`.
    pub fn into_parts(self) -> (Shape, GnaBuffer) {
        (self.shape, self.buffer)
    }

    /// Return a new primitive with the given shape.
    pub fn with_shape(self, shape: Shape) -> Self {
        Self { shape, ..self }
    }
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
