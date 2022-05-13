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
from oneflow.test_utils.automated_test_util import *
import numpy as np
import oneflow as flow
from oneflow.test_utils.test_util import GenArgList



def exp_reducer(x):
    return 3*x

# def exp_adder(x, y):
#     return 2*x+3*y

def exp_tensor_cos(x):
    return x.cos()

def exp_tensor_sin(x):
    return x.sin()



def exp_mul_num(x):
    return torch.mul(x,10)

def exp_mul(x,y):
    return torch.mul(x,y)


@flow.unittest.skip_unless_1n1d()
class TestVjp(flow.unittest.TestCase):
    @autotest(n=5, check_graph=False)
    def test_vjp_exp_reducer_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 2).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func=exp_reducer
        v=torch.ones_like(x)
        create_graph=random().to(bool)
        strict=random().to(bool)
        y = torch.autograd.functional.vjp(func, x, v, create_graph=create_graph, strict=strict)
        
        return y
    
    
    @autotest(n=5, check_graph=False)
    def test_vjp_sin_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func=torch.sin
        v=torch.ones_like(x)
        create_graph=random().to(bool)
        strict=random().to(bool)
        y = torch.autograd.functional.vjp(func, x, v, create_graph=create_graph, strict=strict)
        
        return y


    @autotest(n=5, check_graph=False)
    def test_vjp_tensor_sin_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func=exp_tensor_sin
        v=torch.ones_like(x)
        create_graph=random().to(bool)
        strict=random().to(bool)
        y = torch.autograd.functional.vjp(func, x, v, create_graph=create_graph, strict=strict)
        
        return y
    

    @autotest(n=5, check_graph=False)
    def test_vjp_cos_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func=torch.cos
        v=torch.ones_like(x)
        create_graph=random().to(bool)
        strict=random().to(bool)
        y = torch.autograd.functional.vjp(func, x, v, create_graph=create_graph, strict=strict)
        
        return y    


    @autotest(n=5, check_graph=False)
    def test_vjp_tensor_cos_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func=exp_tensor_cos
        v=torch.ones_like(x)
        create_graph=random().to(bool)
        strict=random().to(bool)
        y = torch.autograd.functional.vjp(func, x, v, create_graph=create_graph, strict=strict)
        
        return y  
    
    
    '''
    #issue:pytorch_tensor
    @autotest(n=1, check_graph=False)
    def test_vjp_tanh_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        def exp_nn_tanh(x):
            m=torch.nn.Tanh()
            m.train(random())
            device = random_device()
            m.to(device)
            #x = random_tensor().to(device)
            #print("x:",type(x))
            return m(x)
        func=exp_nn_tanh
        v=torch.ones_like(x)
        create_graph=random().to(bool)
        strict=random().to(bool)
        y = torch.autograd.functional.vjp(func, x, v, create_graph=create_graph, strict=strict)
        
        return y
    '''
    '''
    @autotest(n=1, check_graph=False)
    def test_vjp_mul_num_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to('cpu')
        func=exp_mul_num
        v=torch.ones_like(x)
        create_graph=random().to(bool)
        strict=random().to(bool)
        y = torch.autograd.functional.vjp(func, x, v, create_graph=create_graph, strict=strict)
        
        return y
    
    @autotest(n=1, check_graph=False)
    def test_vjp_mul_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = (random_tensor(ndim=ndim, requires_grad=True).to(device),random_tensor(ndim=ndim, requires_grad=True).to('cpu'))
        func=exp_mul
        v=torch.ones_like(random_tensor(ndim=ndim, requires_grad=True).to(device))
        create_graph=random().to(bool)
        strict=random().to(bool)
        y = torch.autograd.functional.vjp(func, x, v, create_graph=create_graph, strict=strict)
        
        return y
    '''


if __name__ == "__main__":
    unittest.main()