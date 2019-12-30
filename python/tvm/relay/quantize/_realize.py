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
#pylint: disable=unused-argument,inconsistent-return-statements
"""Internal module for realizing the quantized operations."""
from __future__ import absolute_import

import topi
from ..._ffi.function import register_func
from ... import make as _make
from .. import expr as _expr
from .. import analysis as _analysis
from .. import op as _op
from ..op import op as _reg
from ..base import register_relay_node
from . import _quantize
from .quantize import QAnnotateKind, current_qconfig, quantize_context
from .quantize import _forward_op

def register_realize_function(op_name, frewrite=None, level=10):
    def _register(func):
        return _reg._Register(op_name, "FQRealizeRewrite", func, level)
    return _register(frewrite) if frewrite is not None else _register

    
@register_realize_function("nn.conv2d")
def conv2d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for conv2d. Lhs of conv will be quantized to
    input field, and rhs of conv will be quantized to weight field.
    Output would be in activation field"""
    realize_qconfig = quantize_context().current_realize_qconfig(ref_call)

    with realize_qconfig:
        qnode =  _quantize.conv2d_realize(ref_call, new_args, ctx)

    return qnode
