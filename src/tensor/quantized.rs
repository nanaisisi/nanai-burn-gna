use crate::device::GnaDevice;
use crate::tensor::buffer::GnaBuffer;
use burn_backend::quantization::QuantScheme;
use burn_backend::{DType, Shape, TensorMetadata};

/// Quantized tensor primitive for GnaBackend.
#[derive(Clone, Debug)]
pub struct GnaQTensorPrimitive {
    shape: Shape,
    scheme: QuantScheme,
    buffer: GnaBuffer,
}

impl GnaQTensorPrimitive {
    /// Create a new quantized tensor primitive.
    pub fn new(shape: Shape, scheme: QuantScheme, buffer: GnaBuffer) -> Self {
        Self {
            shape,
            scheme,
            buffer,
        }
    }

    /// Return the quantization scheme.
    pub fn scheme(&self) -> QuantScheme {
        self.scheme
    }

    /// Return a reference to the underlying buffer.
    pub fn buffer(&self) -> &GnaBuffer {
        &self.buffer
    }

    /// Consume the primitive and return its underlying buffer.
    pub fn into_buffer(self) -> GnaBuffer {
        self.buffer
    }

    /// Consume the primitive and return `(shape, scheme, buffer)`.
    pub fn into_parts(self) -> (Shape, QuantScheme, GnaBuffer) {
        (self.shape, self.scheme, self.buffer)
    }

    /// Return a new primitive with the given shape.
    pub fn with_shape(self, shape: Shape) -> Self {
        Self { shape, ..self }
    }

    /// Return the default quantization scheme for GNA.
    pub fn default_scheme() -> QuantScheme {
        GnaDevice::default_quant_scheme()
    }
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
