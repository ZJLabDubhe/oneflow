"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from collections import OrderedDict
#from oneflow.test_utils.automated_test_util import *
import numpy as np
import oneflow as flow
from oneflow.test_utils.test_util import GenArgList


def exp_reducer(x):
    return 3*x

def adder(x, y):
    return 2*x+3*y

def exp_cos(x):
    return flow.cos(x)

def exp_tensor_cos(x):
    return x.cos()

def exp_sin(x):
    return flow.sin(x)

def exp_tensor_sin(x):
    return x.sin()

def exp_nn_tanh(x):
    m=flow.nn.Tanh()
    return m(x)
#pow




def _test_vjp(test_case, device, func, create_graph, strict):
   
    input = flow.tensor(
        np.random.randn(2,3), dtype=flow.float32, device=flow.device(device)
    )
    v = flow.ones(2,3)
    y=flow.autograd.vjp(func, input, v,create_graph=create_graph, strict=strict)
    

def _test_vjp_two_inputs(test_case, device, func, create_graph, strict):
   
    inputs = (flow.tensor(
        np.random.randn(2,3), dtype=flow.float32, device=flow.device(device)
    ),flow.tensor(
        np.random.randn(2,3), dtype=flow.float32, device=flow.device(device)
    ))
    v = flow.ones(2,3)

    y=flow.autograd.vjp(func, inputs, v,create_graph=create_graph, strict=strict)


@flow.unittest.skip_unless_1n1d()
class TestVjp(flow.unittest.TestCase):
    def test_vjp(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_vjp,
            '''_test_vjp_two_inputs,'''

        ]
        arg_dict["device"] = ["cpu"]
        arg_dict["func"] = [
            exp_reducer, 
            exp_cos,
            exp_tensor_cos, 
            exp_sin, 
            exp_tensor_sin,
            exp_nn_tanh,

        ]
        arg_dict["create_graph "] = [False, True]
        arg_dict["strict "] = [False]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()