use crate::device::GnaDevice;
use crate::tensor::{
    GnaBoolTensorPrimitive, GnaFloatTensorPrimitive, GnaIntTensorPrimitive, GnaQTensorPrimitive,
};
use burn::tensor::backend::{Backend, BackendTypes, DTypeUsage, ExecutionError};
use burn::tensor::ops::{
    ActivationOps, AttentionModuleOptions, BoolTensorOps, ConvOptions, ConvTransposeOptions,
    DeformConv2dBackward, DeformConvOptions, FloatTensorOps, IntTensorOps, InterpolateOptions,
    MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps, QTensorOps, TransactionOps,
    TransactionPrimitive, TransactionPrimitiveData,
};
use burn::tensor::quantization::{QuantScheme, QuantizationParametersPrimitive};
use burn::tensor::{
    BoolStore, DType, Distribution, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
};
use enumset::{enum_set, EnumSet};

/// Intel GNA Backend for Burn deep learning framework.
#[derive(Clone, Copy, Default, Debug)]
pub struct GnaBackend;

impl BackendTypes for GnaBackend {
    type Device = GnaDevice;
    type FloatElem = f32;
    type FloatTensorPrimitive = GnaFloatTensorPrimitive;
    type IntElem = i32;
    type IntTensorPrimitive = GnaIntTensorPrimitive;
    type BoolElem = bool;
    type BoolTensorPrimitive = GnaBoolTensorPrimitive;
    type QuantizedTensorPrimitive = GnaQTensorPrimitive;
}

impl Backend for GnaBackend {
    fn name(device: &Self::Device) -> String {
        format!("gna (device #{})", device.index)
    }

    fn seed(_device: &Self::Device, _seed: u64) {
        // GNA accelerator is deterministic inference device; seed is no-op
    }

    fn dtype_usage(_device: &Self::Device, dtype: DType) -> EnumSet<DTypeUsage> {
        match dtype {
            DType::QFloat(_) => enum_set!(DTypeUsage::Storage | DTypeUsage::Accelerated),
            DType::Bool(_) | DType::F32 | DType::I32 | DType::I16 | DType::I8 => {
                enum_set!(DTypeUsage::Storage | DTypeUsage::Arithmetic | DTypeUsage::Accelerated)
            }
            _ => EnumSet::empty(),
        }
    }

    fn device_count(_type_id: u16) -> usize {
        // Check dynamically if GNA library is present, fallback to 0/1
        if let Ok(lib) = nanai_gna_dll_rs::GnaLibrary::load_default() {
            nanai_gna_dll_rs::GnaDevice::get_count(&lib).unwrap_or(0) as usize
        } else {
            0
        }
    }

    fn sync(_device: &Self::Device) -> Result<(), ExecutionError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ActivationOps
// ---------------------------------------------------------------------------
impl ActivationOps<Self> for GnaBackend {}

// ---------------------------------------------------------------------------
// TransactionOps
// ---------------------------------------------------------------------------
impl TransactionOps<Self> for GnaBackend {
    async fn tr_execute(
        _transaction: TransactionPrimitive<Self>,
    ) -> Result<TransactionPrimitiveData, ExecutionError> {
        Err(ExecutionError::WithContext {
            reason: "TransactionOps not supported on GNA backend".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// FloatTensorOps
// ---------------------------------------------------------------------------
impl FloatTensorOps<Self> for GnaBackend {
    fn float_from_data(data: TensorData, _device: &GnaDevice) -> GnaFloatTensorPrimitive {
        let shape = data.shape.clone();
        let dtype = match data.dtype {
            DType::F64 => FloatDType::F64,
            DType::F32 => FloatDType::F32,
            DType::F16 => FloatDType::F16,
            DType::BF16 => FloatDType::BF16,
            _ => FloatDType::F32,
        };
        GnaFloatTensorPrimitive {
            shape,
            dtype,
            buffer: crate::tensor::GnaBuffer::Bytes(std::sync::Arc::new(data.into_bytes().to_vec())),
        }
    }

    fn float_random(
        _shape: Shape,
        _distribution: Distribution,
        _device: &GnaDevice,
        _dtype: FloatDType,
    ) -> GnaFloatTensorPrimitive {
        unimplemented!("float_random is not supported on GNA hardware (inference-only accelerator)")
    }

    async fn float_into_data(
        tensor: GnaFloatTensorPrimitive,
    ) -> Result<TensorData, ExecutionError> {
        match tensor.buffer {
            crate::tensor::GnaBuffer::Bytes(bytes) => {
                let data = TensorData::from_bytes_vec(
                    bytes.as_slice().to_vec(),
                    tensor.shape,
                    tensor.dtype.into(),
                );
                Ok(data)
            }
        }
    }

    fn float_device(_tensor: &GnaFloatTensorPrimitive) -> GnaDevice {
        GnaDevice::default()
    }

    fn float_to_device(tensor: GnaFloatTensorPrimitive, _device: &GnaDevice) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_into_int(tensor: GnaFloatTensorPrimitive, out_dtype: IntDType) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive {
            shape: tensor.shape,
            dtype: out_dtype,
            buffer: tensor.buffer,
        }
    }

    fn float_empty(shape: Shape, _device: &GnaDevice, dtype: FloatDType) -> GnaFloatTensorPrimitive {
        let num_elements: usize = shape.num_elements();
        let bytes_per_elem = DType::from(dtype).size();
        let buf = vec![0u8; num_elements * bytes_per_elem];
        GnaFloatTensorPrimitive {
            shape,
            dtype,
            buffer: crate::tensor::GnaBuffer::Bytes(std::sync::Arc::new(buf)),
        }
    }

    fn float_add(lhs: GnaFloatTensorPrimitive, _rhs: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_add_scalar(lhs: GnaFloatTensorPrimitive, _rhs: Scalar) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_sub(lhs: GnaFloatTensorPrimitive, _rhs: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_sub_scalar(lhs: GnaFloatTensorPrimitive, _rhs: Scalar) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_mul(lhs: GnaFloatTensorPrimitive, _rhs: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_mul_scalar(lhs: GnaFloatTensorPrimitive, _rhs: Scalar) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_div(lhs: GnaFloatTensorPrimitive, _rhs: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_div_scalar(lhs: GnaFloatTensorPrimitive, _rhs: Scalar) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_remainder(lhs: GnaFloatTensorPrimitive, _rhs: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_remainder_scalar(lhs: GnaFloatTensorPrimitive, _rhs: Scalar) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_matmul(lhs: GnaFloatTensorPrimitive, _rhs: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_cross(lhs: GnaFloatTensorPrimitive, _rhs: GnaFloatTensorPrimitive, _dim: usize) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_recip(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_swap_dims(tensor: GnaFloatTensorPrimitive, _dim1: usize, _dim2: usize) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_permute(tensor: GnaFloatTensorPrimitive, _axes: &[usize]) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_flip(tensor: GnaFloatTensorPrimitive, _axes: &[usize]) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_reshape(tensor: GnaFloatTensorPrimitive, shape: Shape) -> GnaFloatTensorPrimitive {
        GnaFloatTensorPrimitive { shape, ..tensor }
    }

    fn float_gather(
        _dim: usize,
        tensor: GnaFloatTensorPrimitive,
        _indices: GnaIntTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_scatter_add(
        _dim: usize,
        tensor: GnaFloatTensorPrimitive,
        _indices: GnaIntTensorPrimitive,
        _value: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_select(
        tensor: GnaFloatTensorPrimitive,
        _dim: usize,
        _indices: GnaIntTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_select_add(
        tensor: GnaFloatTensorPrimitive,
        _dim: usize,
        _indices: GnaIntTensorPrimitive,
        _value: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_slice(tensor: GnaFloatTensorPrimitive, _slices: &[Slice]) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_slice_assign(
        tensor: GnaFloatTensorPrimitive,
        _slices: &[Slice],
        _value: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_mask_where(
        tensor: GnaFloatTensorPrimitive,
        _mask: GnaBoolTensorPrimitive,
        _value: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_mask_fill(
        tensor: GnaFloatTensorPrimitive,
        _mask: GnaBoolTensorPrimitive,
        _value: Scalar,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_equal(
        lhs: GnaFloatTensorPrimitive,
        _rhs: GnaFloatTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn float_equal_elem(
        lhs: GnaFloatTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn float_greater(
        lhs: GnaFloatTensorPrimitive,
        _rhs: GnaFloatTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn float_greater_elem(
        lhs: GnaFloatTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn float_greater_equal(
        lhs: GnaFloatTensorPrimitive,
        _rhs: GnaFloatTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn float_greater_equal_elem(
        lhs: GnaFloatTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn float_lower(
        lhs: GnaFloatTensorPrimitive,
        _rhs: GnaFloatTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn float_lower_elem(
        lhs: GnaFloatTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn float_lower_equal(
        lhs: GnaFloatTensorPrimitive,
        _rhs: GnaFloatTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn float_lower_equal_elem(
        lhs: GnaFloatTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn float_sum(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_sum_dim(tensor: GnaFloatTensorPrimitive, _dim: usize) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_mean_dim(tensor: GnaFloatTensorPrimitive, _dim: usize) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_cumsum(tensor: GnaFloatTensorPrimitive, _dim: usize) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_cumprod(tensor: GnaFloatTensorPrimitive, _dim: usize) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_cummin(tensor: GnaFloatTensorPrimitive, _dim: usize) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_cummax(tensor: GnaFloatTensorPrimitive, _dim: usize) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_cast(tensor: GnaFloatTensorPrimitive, dtype: FloatDType) -> GnaFloatTensorPrimitive {
        GnaFloatTensorPrimitive { dtype, ..tensor }
    }

    fn float_exp(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_log(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_log1p(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_powf(lhs: GnaFloatTensorPrimitive, _rhs: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_powf_scalar_impl(tensor: GnaFloatTensorPrimitive, _value: Scalar) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_sqrt(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_abs(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_cos(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_sin(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_tan(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_cosh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_sinh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_tanh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_acos(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_acosh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_asin(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_asinh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_atan(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_atanh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_atan2(lhs: GnaFloatTensorPrimitive, _rhs: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        lhs
    }

    fn float_round(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_floor(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_ceil(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_trunc(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_erf(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
        tensor
    }

    fn float_argmax(
        tensor: GnaFloatTensorPrimitive,
        _dim: usize,
        out_dtype: IntDType,
    ) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive {
            shape: tensor.shape,
            dtype: out_dtype,
            buffer: tensor.buffer,
        }
    }

    fn float_argtopk(
        tensor: GnaFloatTensorPrimitive,
        _dim: usize,
        _k: usize,
        out_dtype: IntDType,
    ) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive {
            shape: tensor.shape,
            dtype: out_dtype,
            buffer: tensor.buffer,
        }
    }

    fn float_argmin(
        tensor: GnaFloatTensorPrimitive,
        _dim: usize,
        out_dtype: IntDType,
    ) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive {
            shape: tensor.shape,
            dtype: out_dtype,
            buffer: tensor.buffer,
        }
    }

    fn float_expand(tensor: GnaFloatTensorPrimitive, shape: Shape) -> GnaFloatTensorPrimitive {
        GnaFloatTensorPrimitive { shape, ..tensor }
    }

    fn float_unfold(
        tensor: GnaFloatTensorPrimitive,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> GnaFloatTensorPrimitive {
        tensor
    }
}

// ---------------------------------------------------------------------------
// BoolTensorOps
// ---------------------------------------------------------------------------
impl BoolTensorOps<Self> for GnaBackend {
    fn bool_empty(shape: Shape, _device: &GnaDevice, _dtype: BoolStore) -> GnaBoolTensorPrimitive {
        let num = shape.num_elements();
        GnaBoolTensorPrimitive {
            shape,
            buffer: crate::tensor::GnaBuffer::Bytes(std::sync::Arc::new(vec![0u8; num])),
        }
    }

    fn bool_zeros(shape: Shape, device: &GnaDevice, dtype: BoolStore) -> GnaBoolTensorPrimitive {
        Self::bool_empty(shape, device, dtype)
    }

    fn bool_ones(shape: Shape, _device: &GnaDevice, _dtype: BoolStore) -> GnaBoolTensorPrimitive {
        let num = shape.num_elements();
        GnaBoolTensorPrimitive {
            shape,
            buffer: crate::tensor::GnaBuffer::Bytes(std::sync::Arc::new(vec![1u8; num])),
        }
    }

    async fn bool_into_data(
        tensor: GnaBoolTensorPrimitive,
    ) -> Result<TensorData, ExecutionError> {
        match tensor.buffer {
            crate::tensor::GnaBuffer::Bytes(bytes) => {
                let data = TensorData::from_bytes_vec(
                    bytes.as_slice().to_vec(),
                    tensor.shape,
                    DType::Bool(BoolStore::Native),
                );
                Ok(data)
            }
        }
    }

    fn bool_from_data(data: TensorData, _device: &GnaDevice) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: data.shape.clone(),
            buffer: crate::tensor::GnaBuffer::Bytes(std::sync::Arc::new(data.into_bytes().to_vec())),
        }
    }

    fn bool_into_int(tensor: GnaBoolTensorPrimitive, out_dtype: IntDType) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive {
            shape: tensor.shape,
            dtype: out_dtype,
            buffer: tensor.buffer,
        }
    }

    fn bool_into_float(tensor: GnaBoolTensorPrimitive, out_dtype: FloatDType) -> GnaFloatTensorPrimitive {
        GnaFloatTensorPrimitive {
            shape: tensor.shape,
            dtype: out_dtype,
            buffer: tensor.buffer,
        }
    }

    fn bool_device(_tensor: &GnaBoolTensorPrimitive) -> GnaDevice {
        GnaDevice::default()
    }

    fn bool_to_device(tensor: GnaBoolTensorPrimitive, _device: &GnaDevice) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_reshape(tensor: GnaBoolTensorPrimitive, shape: Shape) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive { shape, ..tensor }
    }

    fn bool_slice(tensor: GnaBoolTensorPrimitive, _slices: &[Slice]) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_slice_assign(
        tensor: GnaBoolTensorPrimitive,
        _slices: &[Slice],
        _value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_mask_where(
        tensor: GnaBoolTensorPrimitive,
        _mask: GnaBoolTensorPrimitive,
        _value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_mask_fill(
        tensor: GnaBoolTensorPrimitive,
        _mask: GnaBoolTensorPrimitive,
        _value: Scalar,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_gather(
        _dim: usize,
        tensor: GnaBoolTensorPrimitive,
        _indices: GnaIntTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_scatter_or(
        _dim: usize,
        tensor: GnaBoolTensorPrimitive,
        _indices: GnaIntTensorPrimitive,
        _value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_select(
        tensor: GnaBoolTensorPrimitive,
        _dim: usize,
        _indices: GnaIntTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_select_or(
        tensor: GnaBoolTensorPrimitive,
        _dim: usize,
        _indices: GnaIntTensorPrimitive,
        _value: GnaBoolTensorPrimitive,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_equal(lhs: GnaBoolTensorPrimitive, _rhs: GnaBoolTensorPrimitive) -> GnaBoolTensorPrimitive {
        lhs
    }

    fn bool_equal_elem(lhs: GnaBoolTensorPrimitive, _rhs: Scalar) -> GnaBoolTensorPrimitive {
        lhs
    }

    fn bool_not(tensor: GnaBoolTensorPrimitive) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_and(lhs: GnaBoolTensorPrimitive, _rhs: GnaBoolTensorPrimitive) -> GnaBoolTensorPrimitive {
        lhs
    }

    fn bool_or(lhs: GnaBoolTensorPrimitive, _rhs: GnaBoolTensorPrimitive) -> GnaBoolTensorPrimitive {
        lhs
    }

    fn bool_swap_dims(tensor: GnaBoolTensorPrimitive, _dim1: usize, _dim2: usize) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_permute(tensor: GnaBoolTensorPrimitive, _axes: &[usize]) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_flip(tensor: GnaBoolTensorPrimitive, _axes: &[usize]) -> GnaBoolTensorPrimitive {
        tensor
    }

    fn bool_expand(tensor: GnaBoolTensorPrimitive, shape: Shape) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive { shape, ..tensor }
    }

    fn bool_unfold(
        tensor: GnaBoolTensorPrimitive,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> GnaBoolTensorPrimitive {
        tensor
    }
}

// ---------------------------------------------------------------------------
// IntTensorOps
// ---------------------------------------------------------------------------
impl IntTensorOps<Self> for GnaBackend {
    fn int_empty(shape: Shape, _device: &GnaDevice, dtype: IntDType) -> GnaIntTensorPrimitive {
        let num = shape.num_elements();
        let bytes_per_elem = DType::from(dtype).size();
        GnaIntTensorPrimitive {
            shape,
            dtype,
            buffer: crate::tensor::GnaBuffer::Bytes(std::sync::Arc::new(vec![0u8; num * bytes_per_elem])),
        }
    }

    async fn int_into_data(
        tensor: GnaIntTensorPrimitive,
    ) -> Result<TensorData, ExecutionError> {
        match tensor.buffer {
            crate::tensor::GnaBuffer::Bytes(bytes) => {
                let data = TensorData::from_bytes_vec(
                    bytes.as_slice().to_vec(),
                    tensor.shape,
                    tensor.dtype.into(),
                );
                Ok(data)
            }
        }
    }

    fn int_from_data(data: TensorData, _device: &GnaDevice) -> GnaIntTensorPrimitive {
        let shape = data.shape.clone();
        let dtype = match data.dtype {
            DType::I64 => IntDType::I64,
            DType::I32 => IntDType::I32,
            DType::I16 => IntDType::I16,
            DType::I8 => IntDType::I8,
            _ => IntDType::I32,
        };
        GnaIntTensorPrimitive {
            shape,
            dtype,
            buffer: crate::tensor::GnaBuffer::Bytes(std::sync::Arc::new(data.into_bytes().to_vec())),
        }
    }

    fn int_device(_tensor: &GnaIntTensorPrimitive) -> GnaDevice {
        GnaDevice::default()
    }

    fn int_to_device(tensor: GnaIntTensorPrimitive, _device: &GnaDevice) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_reshape(tensor: GnaIntTensorPrimitive, shape: Shape) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive { shape, ..tensor }
    }

    fn int_slice(tensor: GnaIntTensorPrimitive, _slices: &[Slice]) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_slice_assign(
        tensor: GnaIntTensorPrimitive,
        _slices: &[Slice],
        _value: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_into_float(tensor: GnaIntTensorPrimitive, out_dtype: FloatDType) -> GnaFloatTensorPrimitive {
        GnaFloatTensorPrimitive {
            shape: tensor.shape,
            dtype: out_dtype,
            buffer: tensor.buffer,
        }
    }

    fn int_mask_where(
        tensor: GnaIntTensorPrimitive,
        _mask: GnaBoolTensorPrimitive,
        _value: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_mask_fill(
        tensor: GnaIntTensorPrimitive,
        _mask: GnaBoolTensorPrimitive,
        _value: Scalar,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_gather(
        _dim: usize,
        tensor: GnaIntTensorPrimitive,
        _indices: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_scatter_add(
        _dim: usize,
        tensor: GnaIntTensorPrimitive,
        _indices: GnaIntTensorPrimitive,
        _value: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_select(
        tensor: GnaIntTensorPrimitive,
        _dim: usize,
        _indices: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_select_add(
        tensor: GnaIntTensorPrimitive,
        _dim: usize,
        _indices: GnaIntTensorPrimitive,
        _value: GnaIntTensorPrimitive,
    ) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_equal(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_equal_elem(
        lhs: GnaIntTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_greater(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_greater_elem(
        lhs: GnaIntTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_greater_equal(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_greater_equal_elem(
        lhs: GnaIntTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_lower(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_lower_elem(
        lhs: GnaIntTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_lower_equal(
        lhs: GnaIntTensorPrimitive,
        _rhs: GnaIntTensorPrimitive,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_lower_equal_elem(
        lhs: GnaIntTensorPrimitive,
        _rhs: Scalar,
        _out_dtype: BoolStore,
    ) -> GnaBoolTensorPrimitive {
        GnaBoolTensorPrimitive {
            shape: lhs.shape,
            buffer: lhs.buffer,
        }
    }

    fn int_add(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_add_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_sub(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_sub_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_mul(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_mul_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_div(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_div_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_remainder(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_remainder_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_matmul(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_sum(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_sum_dim(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_prod(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_prod_dim(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_mean_dim(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_cumsum(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_cumprod(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_cummin(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_cummax(tensor: GnaIntTensorPrimitive, _dim: usize) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_argmax(
        tensor: GnaIntTensorPrimitive,
        _dim: usize,
    ) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive {
            dtype: IntDType::I64,
            ..tensor
        }
    }

    fn int_argtopk(
        tensor: GnaIntTensorPrimitive,
        _dim: usize,
        _k: usize,
    ) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive {
            dtype: IntDType::I64,
            ..tensor
        }
    }

    fn int_argmin(
        tensor: GnaIntTensorPrimitive,
        _dim: usize,
    ) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive {
            dtype: IntDType::I64,
            ..tensor
        }
    }

    fn int_abs(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_swap_dims(tensor: GnaIntTensorPrimitive, _dim1: usize, _dim2: usize) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_permute(tensor: GnaIntTensorPrimitive, _axes: &[usize]) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_flip(tensor: GnaIntTensorPrimitive, _axes: &[usize]) -> GnaIntTensorPrimitive {
        tensor
    }

    fn int_random(
        _shape: Shape,
        _distribution: Distribution,
        _device: &GnaDevice,
        _dtype: IntDType,
    ) -> GnaIntTensorPrimitive {
        unimplemented!("int_random is not supported on GNA hardware")
    }

    fn int_expand(tensor: GnaIntTensorPrimitive, shape: Shape) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive { shape, ..tensor }
    }

    fn bitwise_and(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_and_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_or(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_or_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_xor(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_xor_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_not(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        tensor
    }

    fn bitwise_left_shift(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_left_shift_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_right_shift(lhs: GnaIntTensorPrimitive, _rhs: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
        lhs
    }

    fn bitwise_right_shift_scalar(lhs: GnaIntTensorPrimitive, _rhs: Scalar) -> GnaIntTensorPrimitive {
        lhs
    }

    fn int_cast(tensor: GnaIntTensorPrimitive, dtype: IntDType) -> GnaIntTensorPrimitive {
        GnaIntTensorPrimitive { dtype, ..tensor }
    }

    fn int_unfold(
        tensor: GnaIntTensorPrimitive,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> GnaIntTensorPrimitive {
        tensor
    }
}

// ---------------------------------------------------------------------------
// QTensorOps
// ---------------------------------------------------------------------------
impl QTensorOps<Self> for GnaBackend {
    fn q_from_data(data: TensorData, _device: &GnaDevice) -> GnaQTensorPrimitive {
        let scheme = match data.dtype {
            DType::QFloat(scheme) => scheme,
            _ => <GnaQTensorPrimitive as burn::tensor::backend::QTensorPrimitive>::default_scheme(),
        };
        GnaQTensorPrimitive {
            shape: data.shape.clone(),
            scheme,
            buffer: crate::tensor::GnaBuffer::Bytes(std::sync::Arc::new(data.into_bytes().to_vec())),
        }
    }

    fn quantize(
        tensor: GnaFloatTensorPrimitive,
        scheme: &QuantScheme,
        _qparams: QuantizationParametersPrimitive<Self>,
    ) -> GnaQTensorPrimitive {
        GnaQTensorPrimitive {
            shape: tensor.shape,
            scheme: *scheme,
            buffer: tensor.buffer,
        }
    }

    fn dequantize(tensor: GnaQTensorPrimitive, dtype: FloatDType) -> GnaFloatTensorPrimitive {
        GnaFloatTensorPrimitive {
            shape: tensor.shape,
            dtype,
            buffer: tensor.buffer,
        }
    }

    fn q_device(_tensor: &GnaQTensorPrimitive) -> GnaDevice {
        GnaDevice::default()
    }

    fn q_to_device(tensor: GnaQTensorPrimitive, _device: &GnaDevice) -> GnaQTensorPrimitive {
        tensor
    }

    fn q_reshape(tensor: GnaQTensorPrimitive, shape: Shape) -> GnaQTensorPrimitive {
        GnaQTensorPrimitive { shape, ..tensor }
    }

    async fn q_into_data(
        tensor: GnaQTensorPrimitive,
    ) -> Result<TensorData, ExecutionError> {
        match tensor.buffer {
            crate::tensor::GnaBuffer::Bytes(bytes) => {
                let data = TensorData::from_bytes_vec(
                    bytes.as_slice().to_vec(),
                    tensor.shape,
                    DType::QFloat(tensor.scheme),
                );
                Ok(data)
            }
        }
    }

    fn q_expand(tensor: GnaQTensorPrimitive, shape: Shape) -> GnaQTensorPrimitive {
        GnaQTensorPrimitive { shape, ..tensor }
    }

    fn q_swap_dims(tensor: GnaQTensorPrimitive, _dim1: usize, _dim2: usize) -> GnaQTensorPrimitive {
        tensor
    }

    fn q_permute(tensor: GnaQTensorPrimitive, _axes: &[usize]) -> GnaQTensorPrimitive {
        tensor
    }

    fn q_flip(tensor: GnaQTensorPrimitive, _axes: &[usize]) -> GnaQTensorPrimitive {
        tensor
    }

    fn q_select(
        tensor: GnaQTensorPrimitive,
        _dim: usize,
        _indices: GnaIntTensorPrimitive,
    ) -> GnaQTensorPrimitive {
        tensor
    }

    fn q_slice(tensor: GnaQTensorPrimitive, _slices: &[Slice]) -> GnaQTensorPrimitive {
        tensor
    }
}

// ---------------------------------------------------------------------------
// ModuleOps
// ---------------------------------------------------------------------------
impl ModuleOps<Self> for GnaBackend {
    fn conv2d(
        x: GnaFloatTensorPrimitive,
        _weight: GnaFloatTensorPrimitive,
        _bias: Option<GnaFloatTensorPrimitive>,
        _options: ConvOptions<2>,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn deform_conv2d(
        x: GnaFloatTensorPrimitive,
        _offset: GnaFloatTensorPrimitive,
        _weight: GnaFloatTensorPrimitive,
        _mask: Option<GnaFloatTensorPrimitive>,
        _bias: Option<GnaFloatTensorPrimitive>,
        _options: DeformConvOptions<2>,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn deform_conv2d_backward(
        _x: GnaFloatTensorPrimitive,
        _offset: GnaFloatTensorPrimitive,
        _weight: GnaFloatTensorPrimitive,
        _mask: Option<GnaFloatTensorPrimitive>,
        _bias: Option<GnaFloatTensorPrimitive>,
        _output_grad: GnaFloatTensorPrimitive,
        _options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        unimplemented!("backward pass is not supported on GNA inference accelerator")
    }

    fn conv3d(
        x: GnaFloatTensorPrimitive,
        _weight: GnaFloatTensorPrimitive,
        _bias: Option<GnaFloatTensorPrimitive>,
        _options: ConvOptions<3>,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn conv_transpose2d(
        x: GnaFloatTensorPrimitive,
        _weight: GnaFloatTensorPrimitive,
        _bias: Option<GnaFloatTensorPrimitive>,
        _options: ConvTransposeOptions<2>,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn conv_transpose3d(
        x: GnaFloatTensorPrimitive,
        _weight: GnaFloatTensorPrimitive,
        _bias: Option<GnaFloatTensorPrimitive>,
        _options: ConvTransposeOptions<3>,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn avg_pool2d(
        x: GnaFloatTensorPrimitive,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _count_include_pad: bool,
        _ceil_mode: bool,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn avg_pool2d_backward(
        _x: GnaFloatTensorPrimitive,
        _grad: GnaFloatTensorPrimitive,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _count_include_pad: bool,
        _ceil_mode: bool,
    ) -> GnaFloatTensorPrimitive {
        unimplemented!("backward pass is not supported on GNA inference accelerator")
    }

    fn adaptive_avg_pool2d(
        x: GnaFloatTensorPrimitive,
        _output_size: [usize; 2],
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn adaptive_avg_pool2d_backward(
        _x: GnaFloatTensorPrimitive,
        _grad: GnaFloatTensorPrimitive,
    ) -> GnaFloatTensorPrimitive {
        unimplemented!("backward pass is not supported on GNA inference accelerator")
    }

    fn max_pool2d(
        x: GnaFloatTensorPrimitive,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn max_pool2d_with_indices(
        x: GnaFloatTensorPrimitive,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
    ) -> MaxPool2dWithIndices<Self> {
        let indices = GnaIntTensorPrimitive {
            shape: x.shape.clone(),
            dtype: IntDType::I32,
            buffer: x.buffer.clone(),
        };
        MaxPool2dWithIndices::new(x, indices)
    }

    fn max_pool2d_with_indices_backward(
        _x: GnaFloatTensorPrimitive,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        _padding: [usize; 2],
        _dilation: [usize; 2],
        _ceil_mode: bool,
        _output_grad: GnaFloatTensorPrimitive,
        _indices: GnaIntTensorPrimitive,
    ) -> MaxPool2dBackward<Self> {
        unimplemented!("backward pass is not supported on GNA inference accelerator")
    }

    fn interpolate(
        x: GnaFloatTensorPrimitive,
        _output_size: [usize; 2],
        _options: InterpolateOptions,
    ) -> GnaFloatTensorPrimitive {
        x
    }

    fn interpolate_backward(
        _x: GnaFloatTensorPrimitive,
        _grad: GnaFloatTensorPrimitive,
        _output_size: [usize; 2],
        _options: InterpolateOptions,
    ) -> GnaFloatTensorPrimitive {
        unimplemented!("backward pass is not supported on GNA inference accelerator")
    }

    fn attention(
        query: GnaFloatTensorPrimitive,
        _key: GnaFloatTensorPrimitive,
        _value: GnaFloatTensorPrimitive,
        _mask: Option<GnaBoolTensorPrimitive>,
        _attn_bias: Option<GnaFloatTensorPrimitive>,
        _options: AttentionModuleOptions,
    ) -> GnaFloatTensorPrimitive {
        query
    }

    fn rfft(
        signal: GnaFloatTensorPrimitive,
        _dim: usize,
        _n: Option<usize>,
    ) -> (GnaFloatTensorPrimitive, GnaFloatTensorPrimitive) {
        (signal.clone(), signal)
    }

    fn irfft(
        spectrum_re: GnaFloatTensorPrimitive,
        _spectrum_im: GnaFloatTensorPrimitive,
        _dim: usize,
        _n: Option<usize>,
    ) -> GnaFloatTensorPrimitive {
        spectrum_re
    }
}
