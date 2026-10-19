"""Lazily imported CUDA kernels for :mod:`dlr_dflow.fused_norm`."""

import triton
import triton.language as tl


@triton.jit
def feature_offset(group, offset, GROUPS: tl.constexpr, CHANNELS: tl.constexpr,
                   SPATIAL: tl.constexpr, LANE_CHANNELS_LAST: tl.constexpr,
                   INPUT_CHANNELS_LAST: tl.constexpr):
    # Enumerate each group in the lane tensor's layout for coalesced reads and
    # writes. Primal and coefficient tensors can independently use either
    # layout; all three refer to the same channel/spatial coordinates.
    if LANE_CHANNELS_LAST:
        spatial = offset // (CHANNELS // GROUPS)
        channel = group * (CHANNELS // GROUPS) + offset % (CHANNELS // GROUPS)
    else:
        spatial = offset % SPATIAL
        channel = group * (CHANNELS // GROUPS) + offset // SPATIAL
    if INPUT_CHANNELS_LAST:
        return spatial * CHANNELS + channel
    return channel * SPATIAL + spatial


@triton.jit
def apply_single_row(P, Y, Mean, Rstd, Coefficient, Output,
                     GROUPS: tl.constexpr, GROUP_SIZE: tl.constexpr,
                     CHANNELS: tl.constexpr, SPATIAL: tl.constexpr,
                     P_CHANNELS_LAST: tl.constexpr, Y_CHANNELS_LAST: tl.constexpr,
                     COEFFICIENT_CHANNELS_LAST: tl.constexpr,
                     HAS_COEFFICIENT: tl.constexpr, TRANSPOSE: tl.constexpr,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    group = row % GROUPS
    offset = tl.arange(0, BLOCK)
    valid = offset < GROUP_SIZE
    primal_offset = feature_offset(group, offset, GROUPS, CHANNELS, SPATIAL,
                                   Y_CHANNELS_LAST, P_CHANNELS_LAST)
    lane_offset = (row // GROUPS) * CHANNELS * SPATIAL + feature_offset(
        group, offset, GROUPS, CHANNELS, SPATIAL, Y_CHANNELS_LAST, Y_CHANNELS_LAST)
    p = tl.load(P + primal_offset, valid, other=0)
    mu = tl.load(Mean + group)
    rstd = tl.load(Rstd + group)
    y = tl.load(Y + lane_offset, valid, other=0)
    if HAS_COEFFICIENT:
        coefficient_offset = feature_offset(group, offset, GROUPS, CHANNELS, SPATIAL,
                                             Y_CHANNELS_LAST, COEFFICIENT_CHANNELS_LAST)
        coefficient = tl.load(Coefficient + coefficient_offset, valid, other=0)
        if TRANSPOSE:
            y *= coefficient
    centered = p - mu
    mean_y = tl.sum(y, axis=0) / GROUP_SIZE
    mean_centered_y = tl.sum(centered * y, axis=0) / GROUP_SIZE
    value = rstd * (y - mean_y - centered * (rstd * rstd) * mean_centered_y)
    if HAS_COEFFICIENT and not TRANSPOSE:
        value *= coefficient
    tl.store(Output + lane_offset, value, valid)


@triton.jit
def reduce_partials(P, Y, Mean, Coefficient, Partials,
                    GROUPS: tl.constexpr, GROUP_SIZE: tl.constexpr,
                    CHANNELS: tl.constexpr, SPATIAL: tl.constexpr,
                    P_CHANNELS_LAST: tl.constexpr, Y_CHANNELS_LAST: tl.constexpr,
                    COEFFICIENT_CHANNELS_LAST: tl.constexpr,
                    ROWS: tl.constexpr, TILES: tl.constexpr,
                    HAS_COEFFICIENT: tl.constexpr, TRANSPOSE: tl.constexpr,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    group = row % GROUPS
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    valid = offset < GROUP_SIZE
    primal_offset = feature_offset(group, offset, GROUPS, CHANNELS, SPATIAL,
                                   Y_CHANNELS_LAST, P_CHANNELS_LAST)
    lane_offset = (row // GROUPS) * CHANNELS * SPATIAL + feature_offset(
        group, offset, GROUPS, CHANNELS, SPATIAL, Y_CHANNELS_LAST, Y_CHANNELS_LAST)
    p = tl.load(P + primal_offset, valid, other=0)
    mu = tl.load(Mean + group)
    y = tl.load(Y + lane_offset, valid, other=0)
    if HAS_COEFFICIENT and TRANSPOSE:
        coefficient_offset = feature_offset(group, offset, GROUPS, CHANNELS, SPATIAL,
                                             Y_CHANNELS_LAST, COEFFICIENT_CHANNELS_LAST)
        y *= tl.load(Coefficient + coefficient_offset, valid, other=0)
    centered = p - mu
    sum_y = tl.sum(y, axis=0)
    sum_centered_y = tl.sum(centered * y, axis=0)
    location = row * TILES + tile
    tl.store(Partials + location, sum_y)
    tl.store(Partials + ROWS * TILES + location, sum_centered_y)


@triton.jit
def apply_partials(P, Y, Mean, Rstd, Coefficient, Partials, Output,
                   GROUPS: tl.constexpr, GROUP_SIZE: tl.constexpr,
                   CHANNELS: tl.constexpr, SPATIAL: tl.constexpr,
                   P_CHANNELS_LAST: tl.constexpr, Y_CHANNELS_LAST: tl.constexpr,
                   COEFFICIENT_CHANNELS_LAST: tl.constexpr,
                   ROWS: tl.constexpr, TILES: tl.constexpr,
                   HAS_COEFFICIENT: tl.constexpr, TRANSPOSE: tl.constexpr,
                   BLOCK: tl.constexpr, REDUCTION_BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    group = row % GROUPS
    partial_offset = tl.arange(0, REDUCTION_BLOCK)
    location = row * TILES + partial_offset
    sum_y = tl.sum(tl.load(Partials + location, partial_offset < TILES, other=0), axis=0)
    sum_centered_y = tl.sum(tl.load(Partials + ROWS * TILES + location,
                                  partial_offset < TILES, other=0), axis=0)
    mean_y = sum_y / GROUP_SIZE
    mean_centered_y = sum_centered_y / GROUP_SIZE
    mu = tl.load(Mean + group)
    rstd = tl.load(Rstd + group)
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    valid = offset < GROUP_SIZE
    primal_offset = feature_offset(group, offset, GROUPS, CHANNELS, SPATIAL,
                                   Y_CHANNELS_LAST, P_CHANNELS_LAST)
    lane_offset = (row // GROUPS) * CHANNELS * SPATIAL + feature_offset(
        group, offset, GROUPS, CHANNELS, SPATIAL, Y_CHANNELS_LAST, Y_CHANNELS_LAST)
    p = tl.load(P + primal_offset, valid, other=0)
    y = tl.load(Y + lane_offset, valid, other=0)
    if HAS_COEFFICIENT:
        coefficient_offset = feature_offset(group, offset, GROUPS, CHANNELS, SPATIAL,
                                             Y_CHANNELS_LAST, COEFFICIENT_CHANNELS_LAST)
        coefficient = tl.load(Coefficient + coefficient_offset, valid, other=0)
        if TRANSPOSE:
            y *= coefficient
    value = rstd * (y - mean_y - (p - mu) * (rstd * rstd) * mean_centered_y)
    if HAS_COEFFICIENT and not TRANSPOSE:
        value *= coefficient
    tl.store(Output + lane_offset, value, valid)
