macro_rules! creation_ops {
    () => {
        fn float_from_data(data: TensorData, _device: &GnaDevice) -> GnaFloatTensorPrimitive {
            let shape = data.shape().clone();
            let dtype = match data.dtype() {
                DType::F64 => FloatDType::F64,
                DType::F32 => FloatDType::F32,
                DType::F16 => FloatDType::F16,
                DType::BF16 => FloatDType::BF16,
                _ => FloatDType::F32,
            };
            GnaFloatTensorPrimitive {
                shape,
                dtype,
                buffer: GnaBuffer::Bytes(std::sync::Arc::new(data.into_bytes().to_vec())),
            }
        }

        fn float_random(
            _shape: Shape,
            _distribution: Distribution,
            _device: &GnaDevice,
            _dtype: FloatDType,
        ) -> GnaFloatTensorPrimitive {
            unimplemented!(
                "float_random is not supported on GNA hardware (inference-only accelerator)"
            )
        }

        async fn float_into_data(
            tensor: GnaFloatTensorPrimitive,
        ) -> Result<TensorData, ExecutionError> {
            match tensor.buffer {
                GnaBuffer::Bytes(bytes) => {
                    let data = TensorData::from_bytes_vec(
                        bytes.as_slice().to_vec(),
                        tensor.shape,
                        tensor.dtype.into(),
                    );
                    Ok(data)
                }
            }
        }

        fn float_to_device(
            tensor: GnaFloatTensorPrimitive,
            _device: &GnaDevice,
        ) -> GnaFloatTensorPrimitive {
            tensor
        }

        fn float_into_int(
            tensor: GnaFloatTensorPrimitive,
            out_dtype: IntDType,
        ) -> GnaIntTensorPrimitive {
            GnaIntTensorPrimitive {
                shape: tensor.shape,
                dtype: out_dtype,
                buffer: tensor.buffer,
            }
        }

        fn float_empty(
            shape: Shape,
            _device: &GnaDevice,
            dtype: FloatDType,
        ) -> GnaFloatTensorPrimitive {
            let num_elements: usize = shape.num_elements();
            let bytes_per_elem = DType::from(dtype).size();
            let buf = vec![0u8; num_elements * bytes_per_elem];
            GnaFloatTensorPrimitive {
                shape,
                dtype,
                buffer: GnaBuffer::Bytes(std::sync::Arc::new(buf)),
            }
        }

        fn float_cast(
            tensor: GnaFloatTensorPrimitive,
            dtype: FloatDType,
        ) -> GnaFloatTensorPrimitive {
            GnaFloatTensorPrimitive { dtype, ..tensor }
        }

        fn float_expand(tensor: GnaFloatTensorPrimitive, shape: Shape) -> GnaFloatTensorPrimitive {
            GnaFloatTensorPrimitive { shape, ..tensor }
        }

        fn float_reshape(tensor: GnaFloatTensorPrimitive, shape: Shape) -> GnaFloatTensorPrimitive {
            GnaFloatTensorPrimitive { shape, ..tensor }
        }
    };
}

pub(super) use creation_ops;
