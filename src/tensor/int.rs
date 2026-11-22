use crate::device::GnaDevice;
use crate::tensor::buffer::GnaBuffer;
use burn_backend::{DType, IntDType, Shape, TensorMetadata};

/// Int tensor primitive for GnaBackend.
#[derive(Clone, Debug)]
pub struct GnaIntTensorPrimitive {
    shape: Shape,
    dtype: IntDType,
    buffer: GnaBuffer,
}

impl GnaIntTensorPrimitive {
    /// Create a new int tensor primitive.
    pub fn new(shape: Shape, dtype: IntDType, buffer: GnaBuffer) -> Self {
        Self {
            shape,
            dtype,
            buffer,
        }
    }

    /// Return the integer element data type.
    pub fn dtype(&self) -> IntDType {
        self.dtype
    }

    /// Return a reference to the underlying buffer.
    pub fn buffer(&self) -> &GnaBuffer {
        &self.buffer
    }

    /// Consume the primitive and return its underlying buffer.
    pub fn into_buffer(self) -> GnaBuffer {
        self.buffer
    }

    /// Consume the primitive and return `(shape, dtype, buffer)`.
    pub fn into_parts(self) -> (Shape, IntDType, GnaBuffer) {
        (self.shape, self.dtype, self.buffer)
    }

    /// Return a new primitive with the given shape.
    pub fn with_shape(self, shape: Shape) -> Self {
        Self { shape, ..self }
    }

    /// Return a new primitive with the given int dtype.
    pub fn with_dtype(self, dtype: IntDType) -> Self {
        Self { dtype, ..self }
    }
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
