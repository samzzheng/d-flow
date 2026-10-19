"""Experimental NCHW implicit-GEMM convolution, not used by production solvers."""

import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw(Input, Weight, Output, BATCH: tl.constexpr,
                 CHANNELS: tl.constexpr, HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    # Output channels are deliberately fixed at16 for this bounded experiment.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    output_channels = tl.arange(0, 16)
    spatial_size = HEIGHT * WIDTH
    batch = rows // spatial_size
    spatial = rows % spatial_size
    output_y = spatial // WIDTH
    output_x = spatial % WIDTH
    total_k = CHANNELS * 9
    reduction = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, 16), tl.float32)
    for start in range(tl.cdiv(total_k, BLOCK_K)):
        k = start * BLOCK_K + reduction
        input_channel = k // 9
        kernel_y = (k % 9) // 3
        kernel_x = k % 3
        input_y = output_y[:, None] + kernel_y[None, :] - 1
        input_x = output_x[:, None] + kernel_x[None, :] - 1
        input_offset = ((batch[:, None] * CHANNELS + input_channel[None, :]) * HEIGHT
                        + input_y) * WIDTH + input_x
        valid = ((batch[:, None] < BATCH) & (k[None, :] < total_k)
                 & (input_y >= 0) & (input_y < HEIGHT)
                 & (input_x >= 0) & (input_x < WIDTH))
        a = tl.load(Input + input_offset, valid, other=0)
        b = tl.load(Weight + k[:, None] + output_channels[None, :] * total_k,
                    k[:, None] < total_k, other=0)
        # Keep inputs/outputs FP32. Three TF32 products approximate FP32
        # multiplication without FP16/BF16 input or output conversion.
        accumulator = tl.dot(a, b, accumulator, input_precision="tf32x3")
    output_offset = ((batch[:, None] * 16 + output_channels[None, :]) * spatial_size
                     + spatial[:, None])
    tl.store(Output + output_offset, accumulator, batch[:, None] < BATCH)
