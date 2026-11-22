use crate::backend::gna_exec::extra::map_float_unary;
use crate::tensor::float::GnaFloatTensorPrimitive;

pub(super) fn float_exp(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::exp)
}

pub(super) fn float_log(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::ln)
}

pub(super) fn float_log1p(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::ln_1p)
}

pub(super) fn float_sqrt(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::sqrt)
}

pub(super) fn float_abs(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::abs)
}

pub(super) fn float_cos(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::cos)
}

pub(super) fn float_sin(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::sin)
}

pub(super) fn float_tan(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::tan)
}

pub(super) fn float_cosh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::cosh)
}

pub(super) fn float_sinh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::sinh)
}

pub(super) fn float_tanh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::tanh)
}

pub(super) fn float_acos(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::acos)
}

pub(super) fn float_acosh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::acosh)
}

pub(super) fn float_asin(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::asin)
}

pub(super) fn float_asinh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::asinh)
}

pub(super) fn float_atan(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::atan)
}

pub(super) fn float_atanh(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::atanh)
}

pub(super) fn float_round(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::round)
}

pub(super) fn float_floor(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::floor)
}

pub(super) fn float_ceil(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::ceil)
}

pub(super) fn float_trunc(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, f32::trunc)
}

pub(super) fn float_erf(tensor: GnaFloatTensorPrimitive) -> GnaFloatTensorPrimitive {
    map_float_unary(tensor, |x| {
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let ax = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * ax);
        let y = 1.0
            - (((((1.0614054 * t - 1.453152) * t) + 1.4214137) * t - 0.28449674) * t + 0.2548296)
                * t
                * (-ax * ax).exp();
        sign * y
    })
}
