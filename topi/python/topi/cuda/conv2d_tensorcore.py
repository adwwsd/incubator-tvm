# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
# pylint: disable=no-value-for-parameter
"""conv2d using Tensorcore intrinsics"""

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name
"""Int4 conv2d in NHWC layout"""
import tvm
from tvm import te
from tvm import tir
from tvm import autotvm

from .tensor_intrin import intrin_wmma_load_matrix, intrin_wmma_gemm, intrin_wmma_store_matrix
from .injective import schedule_injective_from_existing
from ..nn.pad import pad
from ..nn.util import get_pad_tuple
from ..util import get_const_tuple, traverse_inline

@autotvm.register_topi_compute("conv2d_nhwc_tensorcore_im2col.cuda")
def conv2d_nhwc_tensorcore_im2col(cfg, data, kernel, stride, padding, dilation, layout, out_dtype='int32'):
    """Convolution operator in NHWC layout for int4 using im2col method.
    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    kernel : tvm..te.Tensor
        4-D with shape [num_filter, filter_height, filter_width, in_channel]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding: int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        layout of data

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm..te.Tensor
        5-D with shape [batch, out_height, out_width, out_channels]
    """
    assert layout == 'NHWC'
    assert data.dtype in ['int4'] and kernel.dtype in ['int4']

    kernel_size_m = 8
    kernel_size_n = 8
    kernel_size_k = 32
    ELE_PER_INT = 8

    batch_size, in_height, in_width, _ = get_const_tuple(data.shape)
    out_channels, kernel_h, kernel_w, in_channels = get_const_tuple(kernel.shape)

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    assert dilation_h == 1 and dilation_w == 1, "Only suppport dilation 1 now"

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (kernel_h, kernel_w))

    # compute graph
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    pad_data = pad(data, pad_before, pad_after, name="pad_data")

    # compute the output shape
    out_height = (in_height - kernel_h + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - kernel_w + pad_left + pad_right) // stride_w + 1

    # Validate the shape
    assert in_channels % ELE_PER_INT == 0
    assert out_channels >= kernel_size_m
    assert in_channels * kernel_h * kernel_w >= kernel_size_k
    assert batch_size * out_height * out_width >= kernel_size_n

    def upalign(x, align):
        return (x + align - 1) // align * align

    # Transform the input and the kernel to im2col format
    im2col_M = upalign(out_channels, kernel_size_m)
    im2col_K = upalign(in_channels * kernel_h * kernel_w, kernel_size_k)
    im2col_N = upalign(batch_size * out_height * out_width, kernel_size_n)

    mm = im2col_M // kernel_size_m
    nn = im2col_N // kernel_size_n
    kk = im2col_K // kernel_size_k

    # im2col kernel
    kernel_im2col = te.compute((im2col_M, im2col_K), lambda i, j: \
                            te.if_then_else(te.all(i < out_channels, j < in_channels * kernel_h * kernel_w),
                            kernel[i, j // (in_channels * kernel_w), (j // in_channels) % kernel_w, j % in_channels], 
                            tir.const(0, 'int4')), name='kernel_im2col')


    # Tranposed im2col
    data_im2col_T = te.compute((im2col_N, im2col_K), lambda i, j: \
                            te.if_then_else(te.all(i < batch_size * out_height * out_width, j < in_channels * kernel_h * kernel_w),
                            pad_data[i // (out_height * out_width), 
                            (i // batch_size // out_width % out_height * stride_h) + j // (in_channels * kernel_w),
                            (i % out_width * stride_w) + (j // in_channels) % kernel_w,
                            j % in_channels],
                            tir.const(0, 'int4')), name='data_im2col_T') 


    # Further pack the data and kernel to better fit the tensor core computation
    A = te.compute((mm, kk, kernel_size_m, kernel_size_k), 
                                    lambda i, j, ii, jj: kernel_im2col[i * kernel_size_m + ii][j * kernel_size_k + jj], name='kernel_im2col_pack')
    B = te.compute((nn, kk, kernel_size_n, kernel_size_k),
                                    lambda i, j, ii, jj: data_im2col_T[i * kernel_size_n + ii][j * kernel_size_k + jj], name='data_im2col_T_pack')

    # GEMM
    k1 = te.reduce_axis((0, kk), name='k1')
    k2 = te.reduce_axis((0, kernel_size_k), name='k2')
    C = te.compute((mm, nn, kernel_size_m, kernel_size_n), 
                    lambda i, j, ii, jj: te.sum((A[i, k1, ii, k2] * B[j, k1, jj, k2]).astype('int32'), axis=[k1, k2]),
                    name='gemm_C')

    # Unpack C
    C_unpack = te.compute((im2col_M, im2col_N), 
                            lambda i, j: C[i // kernel_size_m, j // kernel_size_n, i % kernel_size_m, j % kernel_size_n],
                            name='C_unpack')
                            
    # Transform the result back
    output = te.compute((batch_size, out_height, out_width, out_channels), lambda n, oh, ow, oc: \
                            C_unpack[oc, n * out_width * out_height + oh * out_width + ow].astype(out_dtype), 
                            name='output', tag='conv2d_nhwc_tensorcore_im2col')

    return output

@autotvm.register_topi_schedule("conv2d_nhwc_tensorcore_im2col.cuda")
def schedule_conv2d_nhwc_tensorcore_im2col(cfg, outs):
    """Schedule conv2d NHWC im2col int4 template"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'conv2d_nhwc_tensorcore_im2col':
            _schedule_conv2d_nhwc_tensorcore_im2col(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s

def _schedule_conv2d_nhwc_tensorcore_im2col(cfg, s, output):
    C_unpack = output.op.input_tensors[0]
    C = C_unpack.op.input_tensors[0]
    A, B = C.op.input_tensors
    kernel_im2col = A.op.input_tensors[0]
    data_im2col_T = B.op.input_tensors[0]
    pad_data = data_im2col_T.op.input_tensors[0]

    warp_size = 32
    block_row_warps = 2
    block_col_warps = 4
    warp_row_tiles = 2
    warp_col_tiles = 1
    chunk = 4

    block_x = te.thread_axis('blockIdx.x')
    block_y = te.thread_axis('blockIdx.y')
    block_z = te.thread_axis('blockIdx.z')
    thread_x = te.thread_axis('threadIdx.x')
    thread_y = te.thread_axis('threadIdx.y')
    thread_z = te.thread_axis('threadIdx.z')

    # Schedule padding
    if isinstance(pad_data.op, te.tensor.ComputeOp) and "pad" in pad_data.op.tag:
        s[pad_data].compute_inline()

    # Schedule im2col
    s[data_im2col_T].compute_inline()
    s[kernel_im2col].compute_inline()

    # Schedule pack
    schedule_injective_from_existing(s, A)
    schedule_injective_from_existing(s, B)

    # Schedule the output transformation
    s[C_unpack].compute_inline()
    
    # Handle bias
    if output.op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0].output(0)
    schedule_injective_from_existing(s, output)

    AS = s.cache_read(A, 'shared', [C])
    BS = s.cache_read(B, 'shared', [C])
    AF = s.cache_read(AS, 'wmma.matrix_a', [C])
    BF = s.cache_read(BS, 'wmma.matrix_b', [C])
    CF = s.cache_write(C, 'wmma.accumulator')

    # Schedule GEMM computation
    i, j, kernel_i, kernel_j = s[C].op.axis
    i, ii = s[C].split(i, factor=warp_row_tiles)
    block_i, i = s[C].split(i, factor=block_row_warps)
    j, jj = s[C].split(j, factor=warp_col_tiles)
    block_j, j = s[C].split(j, factor=block_col_warps)
    s[C].reorder(block_i, block_j, i, j, ii, jj, kernel_i, kernel_j)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)
    s[C].bind(i, thread_y)
    s[C].bind(j, thread_z)

    # Schedule local computation
    s[CF].compute_at(s[C], j)
    warp_i, warp_j, _i, _j = s[CF].op.axis
    k, _k = CF.op.reduce_axis
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _i, _j, _k)

    s[AF].compute_at(s[CF], ki)
    s[BF].compute_at(s[CF], ki)

    # Schedule A shared memory
    s[AS].compute_at(s[CF], ko)
    xo, yo, xi, yi = AS.op.axis
    tx, xo = s[AS].split(xo, nparts=block_row_warps)
    ty, yo = s[AS].split(yo, nparts=block_col_warps)
    t = s[AS].fuse(xi, yi)
    to, ti = s[AS].split(t, nparts=warp_size)
    s[AS].bind(tx, thread_y)
    s[AS].bind(ty, thread_z)
    s[AS].bind(to, thread_x)

    # Schedule B shared memory
    s[BS].compute_at(s[CF], ko)
    xo, yo, xi, yi = BS.op.axis
    tx, xo = s[BS].split(xo, nparts=block_row_warps)
    ty, yo = s[BS].split(yo, nparts=block_col_warps)
    t = s[BS].fuse(xi, yi)
    to, ti = s[BS].split(t, nparts=warp_size)
    s[BS].bind(tx, thread_y)
    s[BS].bind(ty, thread_z)
    s[BS].bind(to, thread_x)

    # print(tvm.lower(s, [kernel_im2col.op.input_tensors[0], pad_data.op.input_tensors[0], output], simple_mode=True))

    s[AF].tensorize(AF.op.axis[-2], intrin_wmma_load_matrix((8, 8, 32), 'wmma.matrix_a', 'int4'))
    s[BF].tensorize(BF.op.axis[-2], intrin_wmma_load_matrix((8, 8, 32), 'wmma.matrix_b', 'int4'))
    s[C].tensorize(kernel_i, intrin_wmma_store_matrix((8, 8, 32), 'int32'))
    s[CF].tensorize(_i, intrin_wmma_gemm((8, 8, 32), 'int4'))

    # print(tvm.lower(s, [kernel_im2col.op.input_tensors[0], pad_data.op.input_tensors[0], output], simple_mode=True))

    return s



    