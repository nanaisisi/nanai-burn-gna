use burn_backend::{FloatDType, IntDType, bf16, f16};

pub(crate) fn decode_ints(bytes: &[u8], dtype: IntDType) -> Vec<i64> {
    match dtype {
        IntDType::I8 => bytes.iter().map(|&b| (b as i8) as i64).collect(),
        IntDType::U8 => bytes.iter().map(|&b| b as i64).collect(),
        IntDType::I16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| i16::from_le_bytes(c) as i64)
            .collect(),
        IntDType::U16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| u16::from_le_bytes(c) as i64)
            .collect(),
        IntDType::I32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| i32::from_le_bytes(c) as i64)
            .collect(),
        IntDType::U32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c) as i64)
            .collect(),
        IntDType::I64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| i64::from_le_bytes(c))
            .collect(),
        IntDType::U64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| u64::from_le_bytes(c) as i64)
            .collect(),
    }
}

pub(crate) fn encode_ints(values: &[i64], dtype: IntDType) -> Vec<u8> {
    match dtype {
        IntDType::I8 => values.iter().map(|&v| (v as i8) as u8).collect(),
        IntDType::U8 => values.iter().map(|&v| v as u8).collect(),
        IntDType::I16 => values
            .iter()
            .flat_map(|&v| (v as i16).to_le_bytes())
            .collect(),
        IntDType::U16 => values
            .iter()
            .flat_map(|&v| (v as u16).to_le_bytes())
            .collect(),
        IntDType::I32 => values
            .iter()
            .flat_map(|&v| (v as i32).to_le_bytes())
            .collect(),
        IntDType::U32 => values
            .iter()
            .flat_map(|&v| (v as u32).to_le_bytes())
            .collect(),
        IntDType::I64 => values.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        IntDType::U64 => values
            .iter()
            .flat_map(|&v| (v as u64).to_le_bytes())
            .collect(),
    }
}

pub(crate) fn decode_floats(bytes: &[u8], dtype: FloatDType) -> Vec<f32> {
    match dtype {
        FloatDType::F64 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&c| f64::from_le_bytes(c) as f32)
            .collect(),
        FloatDType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| f16::from_le_bytes(c).to_f32())
            .collect(),
        FloatDType::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| bf16::from_le_bytes(c).to_f32())
            .collect(),
        FloatDType::F32 | FloatDType::Flex32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect(),
    }
}

pub(crate) fn encode_floats(values: &[f32], dtype: FloatDType) -> Vec<u8> {
    match dtype {
        FloatDType::F64 => values
            .iter()
            .flat_map(|&v| (v as f64).to_le_bytes())
            .collect(),
        FloatDType::F16 => values
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect(),
        FloatDType::BF16 => values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect(),
        FloatDType::F32 | FloatDType::Flex32 => {
            values.iter().flat_map(|&v| v.to_le_bytes()).collect()
        }
    }
}
