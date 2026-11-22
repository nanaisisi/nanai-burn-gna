macro_rules! creation_ops {
    () => {
        fn int_empty(shape: Shape, _device: &GnaDevice, dtype: IntDType) -> GnaIntTensorPrimitive {
            let num = shape.num_elements();
            let bytes_per_elem = DType::from(dtype).size();
            GnaIntTensorPrimitive {
                shape,
                dtype,
                buffer: GnaBuffer::Bytes(std::sync::Arc::new(vec![0u8; num * bytes_per_elem])),
            }
        }

        async fn int_into_data(
            tensor: GnaIntTensorPrimitive,
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

        fn int_from_data(data: TensorData, _device: &GnaDevice) -> GnaIntTensorPrimitive {
            let shape = data.shape().clone();
            let dtype = match data.dtype() {
                DType::I64 => IntDType::I64,
                DType::I32 => IntDType::I32,
                DType::I16 => IntDType::I16,
                DType::I8 => IntDType::I8,
                _ => IntDType::I32,
            };
            GnaIntTensorPrimitive {
                shape,
                dtype,
                buffer: GnaBuffer::Bytes(std::sync::Arc::new(data.into_bytes().to_vec())),
            }
        }

        fn int_to_device(
            tensor: GnaIntTensorPrimitive,
            _device: &GnaDevice,
        ) -> GnaIntTensorPrimitive {
            tensor
        }

        fn int_reshape(tensor: GnaIntTensorPrimitive, shape: Shape) -> GnaIntTensorPrimitive {
            GnaIntTensorPrimitive { shape, ..tensor }
        }

        fn int_into_float(
            tensor: GnaIntTensorPrimitive,
            out_dtype: FloatDType,
        ) -> GnaFloatTensorPrimitive {
            GnaFloatTensorPrimitive {
                shape: tensor.shape,
                dtype: out_dtype,
                buffer: tensor.buffer,
            }
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

        fn int_cast(tensor: GnaIntTensorPrimitive, dtype: IntDType) -> GnaIntTensorPrimitive {
            GnaIntTensorPrimitive { dtype, ..tensor }
        }
    };
}

pub(super) use creation_ops;
