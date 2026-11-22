use crate::backend::gna_exec::extra::map_int_unary;
use crate::tensor::int::GnaIntTensorPrimitive;

pub(super) fn int_abs(tensor: GnaIntTensorPrimitive) -> GnaIntTensorPrimitive {
    map_int_unary(tensor, i64::wrapping_abs)
}
