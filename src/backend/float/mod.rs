use crate::backend::GnaBackend;
use crate::device::GnaDevice;
use crate::tensor::{
    GnaBoolTensorPrimitive, GnaBuffer, GnaFloatTensorPrimitive, GnaIntTensorPrimitive,
};
use burn_backend::ops::FloatTensorOps;
use burn_backend::tensor::IndexingUpdateOp;
use burn_backend::{
    BoolStore, DType, Distribution, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice,
    TensorData,
};

mod arithmetic;
mod comparison;
mod creation;
mod indexing;
mod unary;

impl FloatTensorOps<Self> for GnaBackend {
    creation::creation_ops!();
    arithmetic::arithmetic_ops!();
    unary::unary_ops!();
    comparison::comparison_ops!();
    indexing::indexing_ops!();
}
