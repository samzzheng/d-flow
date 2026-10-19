"""Optional pointwise-only primal fusion; normalization statistics stay external."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def pointwise_record(Normalized, Weight, Bias, Scale, Shift, Output, Coefficient,
                     COUNT: tl.constexpr, SPATIAL: tl.constexpr,
                     CONDITIONED: tl.constexpr, FUSE_SILU: tl.constexpr,
                     BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < COUNT
    channel = offset // SPATIAL
    normalized = tl.load(Normalized + offset, valid, other=0)
    weight = tl.load(Weight + channel, valid, other=0)
    bias = tl.load(Bias + channel, valid, other=0)
    preactivation = normalized * weight + bias
    affine = weight
    if CONDITIONED:
        scale = tl.load(Scale + channel, valid, other=0)
        shift = tl.load(Shift + channel, valid, other=0)
        factor = 1 + scale
        preactivation = preactivation * factor + shift
        affine = affine * factor
    if FUSE_SILU:
        denominator = 1 + libdevice.exp(-preactivation)
        sigmoid = tl.div_rn(1., denominator)
        value = tl.div_rn(preactivation, denominator)
        coefficient = affine * sigmoid * (1 + preactivation * (1 - sigmoid))
        tl.store(Coefficient + offset, coefficient, valid)
    else:
        value = preactivation
    tl.store(Output + offset, value, valid)
