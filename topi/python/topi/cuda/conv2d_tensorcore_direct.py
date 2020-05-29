import tvm
from tvm import te
from tvm import autotvm

import numpy as np
from ..util import get_const_tuple, traverse_inline, simplify, tag
from ..nn.pad import pad
from ..nn.util import get_pad_tuple
from .injective import schedule_injective_from_existing
from .tensor_intrin import intrin_wmma_load_matrix_A, intrin_wmma_load_matrix_W, intrin_wmma_store_matrix, intrin_wmma_gemm

def is_shape_tensorcore_direct_qualified(batch, in_channels, num_filter, in_dtype):
    if in_dtype in ['int4', 'uint4']:
        return (batch % 8 == 0 and in_channels % 32 == 0 and num_filter % 8 == 0)
    elif in_dtype in ['int8', 'uint8']:
        return (batch % 16 == 0 and in_channels % 16 == 0 and num_filter % 16 == 0) or \
               (batch % 8 == 0 and in_channels % 16 == 0 and num_filter % 32 == 0) or \
               (batch % 32 == 0 and in_channels % 16 == 0 and num_filter % 8 == 0)
    else:
        return False

def unpack_NHWCnc_to_nhwc(packed_out, out_dtype):
    """Unpack conv2d_NHWC output from layout NHWCnc to NHWC

     Parameters
    -----------
    packed_out : tvm.te.Tensor
        The output tensor of conv2d_NHWC.

    out_dtype : str
        The output dtype.

    Returns
    -------
    unpacked_out : tvm.te.Tensor
        The unpacked output tensor in NHWC layout.
    """
    N, H, W, O, wmma_m, wmma_n = get_const_tuple(packed_out.shape)

    idxmod = tvm.tir.indexmod
    idxdiv = tvm.tir.indexdiv

    oshape = (N * wmma_m, H, W, O * wmma_n)
    unpacked_out = \
        te.compute(oshape,
                   lambda n, h, w, o:
                   packed_out[idxdiv(n, wmma_m), h, w, idxdiv(o, wmma_n), idxmod(n, wmma_m), idxmod(o, wmma_n)]
                   .astype(out_dtype),
                   name='output_unpack',
                   tag=tag.INJECTIVE+",unpack_hwcnc")
    return unpacked_out


def conv2d_nhwc_tensorcore_direct(data, kernel, strides, padding, dilation, out_dtype="int32"):
    """Compute conv2d internally using conv2d_nchwc layout for int8 dtype"""
    assert data.dtype in ('int4', 'uint4', 'int8', 'uint8')
    assert kernel.dtype in ('int4', 'uint4', 'int8', 'uint8')
    # assert data.dtype == kernel.dtype
    packed_out = conv2d_NHWCnc_tensorcore(data, kernel, strides, padding, dilation, out_dtype)
    return unpack_NHWCnc_to_nhwc(packed_out, out_dtype)


def schedule_conv2d_nhwc_tensorcore_direct(outs):
    """Create schedule for tensors"""
    return schedule_NHWCnc_tensorcore(outs)


@autotvm.register_topi_compute("conv2d_NHWCnc_tensorcore.cuda")
def conv2d_NHWCnc_tensorcore(cfg, Input, Filter, stride, padding, dilation, out_dtype="int32"):
    """Compute declaration for tensorcore"""
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    in_dtype = Input.dtype

    if in_dtype in ['int4', 'uint4']:
        wmma_n = wmma_m = 8
        wmma_k = 32
    else:
        wmma_m = 8
        wmma_n = 32
        wmma_k = 16

    pre_computed = len(Filter.shape) == 6
    batch, in_height, in_width, in_channels= get_const_tuple(Input.shape)
    if pre_computed:
        kernel_h, kernel_w, oc_chunk, ic_chunk, oc_block_factor, ic_block_factor = get_const_tuple(Filter.shape)
        num_filter = oc_block_factor * oc_chunk
    else:
        kernel_h, kernel_w, num_filter, _ = get_const_tuple(Filter.shape)

    if in_dtype in ['int4', 'uint4']:
        assert (batch % 8 == 0 and in_channels % 32 == 0 and num_filter % 8 == 0)
    else:
        assert (batch % 16 == 0 and in_channels % 16 == 0 and num_filter % 16 == 0) or \
               (batch % 8 == 0 and in_channels % 16 == 0 and num_filter % 32 == 0) or \
               (batch % 32 == 0 and in_channels % 16 == 0 and num_filter % 8 == 0), \
               "The shape of (batch, in_channels, num_filter) "\
               "must be multiple of (16, 16, 16) or (32, 16, 8) or (8, 16, 32) for fp16 and int8, "\
               "and (8, 32, 8) for int4"

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))

    out_channels = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    cfg.add_flop(2 * batch * out_height * out_width * out_channels * in_channels * kernel_h * kernel_w)

    # Input feature map: (N, H, W, IC, n, ic)
    data_shape = (batch // wmma_m,
                  in_height,
                  in_width,
                  in_channels // wmma_k,
                  wmma_m,
                  wmma_k)

    # Kernel: (H, W, OC, IC, ic, oc)
    kernel_shape = (kernel_h,
                    kernel_w,
                    out_channels // wmma_n,
                    in_channels // wmma_k,
                    wmma_n,
                    wmma_k)

    # Reduction axes
    kh = te.reduce_axis((0, kernel_h), name='kh')
    kw = te.reduce_axis((0, kernel_w), name='kw')
    ic = te.reduce_axis((0, in_channels // wmma_k), name='ic')
    ii = te.reduce_axis((0, wmma_k), name='ii')

    if pre_computed:
        packed_kernel = Filter
    else:
        packed_kernel = te.compute(kernel_shape,
            lambda kh, kw, o, i, oo, ii: Filter[kh, kw, o * wmma_n + oo, i * wmma_k + ii],
            name="packed_kernel"
        )

    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    pad_data = pad(Input, pad_before, pad_after, name="pad_data")

    packed_data = te.compute(data_shape,
        lambda n, h, w, i, nn, ii: pad_data[n * wmma_m + nn, h, w, i * wmma_k + ii]
    )

    Conv = te.compute((batch // wmma_m, out_height, out_width, out_channels // wmma_n, wmma_m, wmma_n),
                    lambda n, h, w, o, nn, oo: te.sum(
                        (packed_data[n, h * stride_h + kh, w * stride_w + kw, ic, nn, ii] *
                        packed_kernel[kh, kw, o, ic, oo, ii]).astype('int32'),
                        axis=[ic, kh, kw, ii]),
                    name="Conv", tag="conv2d_NHWCnc_tensorcore")
    return Conv


def _schedule_NHWCnc_tensorcore(cfg, s, Conv):
    packed_data, packed_kernel = s[Conv].op.input_tensors
    ic, kh, kw, ii = s[Conv].op.reduce_axis
    pad_data = s[packed_data].op.input_tensors[0]

    block_x = te.thread_axis('blockIdx.x')
    block_y = te.thread_axis('blockIdx.y')
    block_z = te.thread_axis('blockIdx.z')
    thread_x = te.thread_axis('threadIdx.x')
    thread_y = te.thread_axis('threadIdx.y')
    thread_z = te.thread_axis('threadIdx.z')

    # Designate the memory hierarchy
    AS = s.cache_read(packed_data, 'shared', [Conv])
    WS = s.cache_read(packed_kernel, 'shared', [Conv])
    AF = s.cache_read(AS, 'wmma.matrix_a', [Conv])
    WF = s.cache_read(WS, 'wmma.matrix_b', [Conv])
    ConvF = s.cache_write(Conv, 'wmma.accumulator')

    if Conv.op in s.outputs:
        output = Conv
        ConvS = s.cache_read(ConvF, 'shared', [Conv])
        OL = ConvS
    else:
        output = s.outputs[0].output(0)
        s[Conv].set_scope('shared')
        OL = Conv

    out_dtype = Conv.dtype

    if isinstance(packed_kernel.op, te.tensor.ComputeOp) and packed_kernel.name == "packed_kernel":
        if autotvm.GLOBAL_SCOPE.in_tuning:
            s[packed_kernel].pragma(s[packed_kernel].op.axis[0], "debug_skip_region")
        else:
                # with tvm.target.create('cuda'):
            schedule_injective_from_existing(s, packed_kernel)

    # Schedule padding
    if isinstance(pad_data.op, te.tensor.ComputeOp) and "pad" in pad_data.op.tag:
        s[pad_data].compute_inline()
        data = pad_data.op.input_tensors[0]

        if autotvm.GLOBAL_SCOPE.in_tuning:
            # skip this part during tuning to make recrods accurate
            # this part will be pre-computed during NNVM's pre-compute optimization pass
            s[pad_data].pragma(s[pad_data].op.axis[0], "debug_skip_region")
    else:
        data = pad_data

    data_dtype = data.dtype
    kernel_dtype = packed_kernel.dtype

    # Schedule for autotvm
    cfg.define_knob("block_row_warps", [1, 2, 4])
    cfg.define_knob("block_col_warps", [1, 2, 4])
    cfg.define_knob("warp_row_tiles", [1, 2, 4, 8, 16])
    cfg.define_knob("warp_col_tiles", [1, 2, 4, 8, 16])
    cfg.define_knob("chunk", [1, 2, 4, 8])
    cfg.define_knob("fuse_pack", [0, 1])
    cfg.define_knob("split_block_k", [1, 2, 4, 8, 16, 32])
    if data_dtype in ["int8", "uint8"]:
        cfg.define_knob("vector_width", [1, 2, 4, 8, 16])
    else:
        cfg.define_knob("vector_width", [4])

    block_row_warps = cfg["block_row_warps"].val
    block_col_warps = cfg["block_col_warps"].val
    warp_row_tiles = cfg["warp_row_tiles"].val
    warp_col_tiles = cfg["warp_col_tiles"].val
    chunk = cfg["chunk"].val
    vector_width = cfg["vector_width"].val
    split_block_k = cfg["split_block_k"].val
    fuse_pack = cfg["fuse_pack"].val

    if not fuse_pack:
        s[packed_data].compute_inline()
    else:
        with tvm.target.create('cuda'):
            schedule_injective_from_existing(s, packed_data)

    if data_dtype in ['int4', 'uint4']:
        wmma_m = wmma_n = 8
        wmma_k = 32
    else:
        # if (batch % 16 == 0 and out_channels % 16 == 0):
        #     cfg.define_knob("wmma_m", [16, 8, 32])
        # elif (batch % 8 == 0 and out_channels % 32 == 0):
        #     cfg.define_knob("wmma_m", [8, 16, 32])
        # elif (batch % 32 == 0 and out_channels % 8 == 0):
        #     cfg.define_knob("wmma_m", [32, 16, 8])
        # wmma_m = cfg["wmma_m"].val
        wmma_m = 8
        wmma_n = 32
        wmma_k = 16
        # if wmma_m == 16:
        #     wmma_n = 16
        # elif wmma_m == 8:
        #     wmma_n = 32
        # elif wmma_m == 32:
        #     wmma_n = 8

    warp_size = 32

    # Schedule for output
    if len(s[output].op.axis) == 4:
        nc, hc, wc, oc = output.op.axis
        nc, nnc = s[output].split(nc, factor=wmma_m)
        oc, ooc = s[output].split(oc, factor=wmma_n)
    else:
        nc, hc, wc, oc, nnc, ooc = output.op.axis

    block_k = s[output].fuse(hc, wc)
    block_k, sub_block_k = s[output].split(block_k, factor=split_block_k)
    nc, nci = s[output].split(nc, factor=warp_row_tiles)
    block_i, nc = s[output].split(nc, factor=block_row_warps)
    oc, oci = s[output].split(oc, factor=warp_col_tiles)
    block_j, oc = s[output].split(oc, factor=block_col_warps)
    s[output].reorder(block_k, sub_block_k, block_i, block_j, nc, oc, nci, oci, nnc, ooc)
    t = s[output].fuse(nnc, ooc)
    _, tx = s[output].split(t, factor=warp_size)
    s[output].bind(block_k, block_z)
    s[output].bind(block_i, block_x)
    s[output].bind(block_j, block_y)
    s[output].bind(tx, thread_x)
    s[output].bind(nc, thread_y)
    s[output].bind(oc, thread_z)

    # Schedule wmma store
    s[OL].compute_at(s[output], block_j)
    nc, hc, wc, oc, nnc, ooc = OL.op.axis
    oc, oci = s[OL].split(oc, factor=warp_col_tiles)
    _, oc = s[OL].split(oc, factor=block_col_warps)
    nc, nci = s[OL].split(nc, factor=warp_row_tiles)
    _, nc = s[OL].split(nc, factor=block_row_warps)
    s[OL].reorder(nc, oc, nci, oci, nnc, ooc)
    s[OL].bind(nc, thread_y)
    s[OL].bind(oc, thread_z)

    # Schedule local computation
    s[ConvF].compute_at(s[OL], oc)
    n, h, w, o, nnf, oof = ConvF.op.axis
    ko, ki = s[ConvF].split(ic, factor=chunk)
    s[ConvF].reorder(ko, kh, ki, kw, n, o, nnf, oof, ii)

    # Move intermediate computation into each output compute tile
    s[AF].compute_at(s[ConvF], kw)
    s[WF].compute_at(s[ConvF], kw)

    # Schedule for A's share memory
    s[AS].compute_at(s[ConvF], kh)
    n, h, w, i, nn, ii = AS.op.axis
    tx, xo = s[AS].split(n, nparts=block_row_warps)
    ty, yo = s[AS].split(xo, nparts=block_col_warps)
    t = s[AS].fuse(nn, ii)
    to, ti = s[AS].split(t, nparts=warp_size)
    ti, _t = s[AS].split(ti, factor=vector_width)
    s[AS].bind(tx, thread_y)
    s[AS].bind(ty, thread_z)
    s[AS].bind(to, thread_x)
    s[AS].vectorize(_t)

    # Schedule for W's share memory
    s[WS].compute_at(s[ConvF], kw)
    kh, kw, ic, o, ii, oo = WS.op.axis
    tx, xo = s[WS].split(o, nparts=block_row_warps)
    ty, yo = s[WS].split(xo, nparts=block_col_warps)
    t = s[WS].fuse(ii, oo)
    to, ti = s[WS].split(t, nparts=warp_size)
    ti, _t = s[WS].split(ti, factor=vector_width)
    s[WS].bind(tx, thread_y)
    s[WS].bind(ty, thread_z)
    s[WS].bind(to, thread_x)
    s[WS].vectorize(ti)

    shape = (wmma_m, wmma_n, wmma_k)

    AS_shape = (wmma_m, wmma_k)
    AL_shape = (wmma_m, wmma_k)
    WS_shape = (wmma_n, wmma_k)
    WL_shape = (wmma_n, wmma_k)
    CL_shape = (wmma_m, wmma_n)
    CS_shape = (wmma_m, wmma_n)

    AL_gemm = te.placeholder(AL_shape, name='A', dtype=data_dtype)
    WL_gemm = te.placeholder(WL_shape, name='B', dtype=kernel_dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k")
    CL_compute = te.compute(CL_shape, lambda ii, jj:
                    te.sum((AL_gemm[ii, k_gemm] * WL_gemm[jj, k_gemm]).astype(out_dtype), axis=k_gemm),
                    name='C')

    AL_strides = [wmma_k, 1]
    AS_strides = [wmma_k, 1]
    WL_strides = [wmma_k, 1]
    WS_strides = [wmma_k, 1]
    CL_strides = [wmma_n, 1]
    CS_strides = [wmma_n, 1]


    s[AF].tensorize(AF.op.axis[-2], intrin_wmma_load_matrix_A(AL_strides, AS_strides, shape,
                                                    "row_major", AS_shape, AL_shape, data_dtype))

    s[WF].tensorize(WF.op.axis[-2], intrin_wmma_load_matrix_W(WL_strides, WS_strides, shape,
                                                    "col_major", WS_shape, WL_shape, kernel_dtype))

    s[OL].tensorize(nnc, intrin_wmma_store_matrix(CS_strides, CL_strides,
                                                  shape, out_dtype, CL_shape, CS_shape, C_scope='shared'))


    s[ConvF].tensorize(nnf, intrin_wmma_gemm(AL_gemm, WL_gemm, CL_compute, AL_strides,
                                             WL_strides, CL_strides, shape))

    # print(tvm.lower(s, [data, packed_kernel, Conv], simple_mode=True))

    return s

@autotvm.register_topi_schedule("conv2d_NHWCnc_tensorcore.cuda")
def schedule_NHWCnc_tensorcore(cfg, outs):
    """TOPI schedule callback"""
    s = te.create_schedule([x.op for x in outs])
    def _callback(op):
        if 'conv2d_NHWCnc_tensorcore' in op.tag:
            _schedule_NHWCnc_tensorcore(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s

