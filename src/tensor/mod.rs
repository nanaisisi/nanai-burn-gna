pub mod bool;
pub mod buffer;
pub mod float;
pub mod int;
pub mod quantized;

pub use bool::GnaBoolTensorPrimitive;
pub use buffer::GnaBuffer;
pub use float::GnaFloatTensorPrimitive;
pub use int::GnaIntTensorPrimitive;
pub use quantized::GnaQTensorPrimitive;
