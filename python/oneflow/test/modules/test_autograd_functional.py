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

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


def _exp_reducer(x):
    return 3 * x * x


def _exp_tensor_cos(x):
    return x.cos()


def _exp_tensor_sin(x):
    return x.sin()


def _exp_tensor_tanh(x):
    return x.tanh()


@flow.unittest.skip_unless_1n1d()
class TestAutogradFunctiona(flow.unittest.TestCase):
    @autotest(n=5, check_graph=False)
    def test_vjp_exp_reducer_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 2).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = _exp_reducer
        v = torch.ones_like(x)
        create_graph = random().to(bool)
        strict = random().to(bool)
        y = torch.autograd.functional.vjp(
            func, x, v, create_graph=create_graph, strict=strict
        )

        return y

    @autotest(n=5, check_graph=False)
    def test_vjp_sin_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = torch.sin
        v = torch.ones_like(x)
        create_graph = random().to(bool)
        strict = random().to(bool)
        y = torch.autograd.functional.vjp(
            func, x, v, create_graph=create_graph, strict=strict
        )

        return y

    @autotest(n=5, check_graph=False)
    def test_vjp_tensor_sin_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = _exp_tensor_sin
        v = torch.ones_like(x)
        create_graph = random().to(bool)
        strict = random().to(bool)
        y = torch.autograd.functional.vjp(
            func, x, v, create_graph=create_graph, strict=strict
        )

        return y

    @autotest(n=5, check_graph=False)
    def test_vjp_cos_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = torch.cos
        v = torch.ones_like(x)
        create_graph = random().to(bool)
        strict = random().to(bool)
        y = torch.autograd.functional.vjp(
            func, x, v, create_graph=create_graph, strict=strict
        )

        return y

    @autotest(n=5, check_graph=False)
    def test_vjp_tensor_cos_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = _exp_tensor_cos
        v = torch.ones_like(x)
        create_graph = random().to(bool)
        strict = random().to(bool)
        y = torch.autograd.functional.vjp(
            func, x, v, create_graph=create_graph, strict=strict
        )

        return y

    @autotest(n=5, check_graph=False)
    def test_vjp_tanh_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = _exp_tensor_tanh
        v = torch.ones_like(x)
        create_graph = random().to(bool)
        strict = random().to(bool)
        y = torch.autograd.functional.vjp(
            func, x, v, create_graph=create_graph, strict=strict
        )

        return y

    @autotest(n=5, check_graph=False)
    def test_vhp_exp_reducer_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 2).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = _exp_reducer
        v = torch.ones_like(x)
        create_graph = random().to(bool)
        strict = random().to(bool)
        y = torch.autograd.functional.vhp(
            func, x, v, create_graph=create_graph, strict=strict
        )

        return y

    @autotest(n=5, check_graph=False)
    def test_vhp_sin_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = torch.sin
        v = torch.ones_like(x)
        create_graph = random().to(bool)
        strict = random().to(bool)
        y = torch.autograd.functional.vhp(
            func, x, v, create_graph=create_graph, strict=strict
        )

        return y

    @autotest(n=5, check_graph=False)
    def test_vhp_tensor_sin_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = _exp_tensor_sin
        v = torch.ones_like(x)
        create_graph = random().to(bool)
        strict = random().to(bool)
        y = torch.autograd.functional.vhp(
            func, x, v, create_graph=create_graph, strict=strict
        )

        return y

    @autotest(n=5, check_graph=False)
    def test_vhp_cos_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = torch.cos
        v = torch.ones_like(x)
        create_graph = random().to(bool)
        strict = random().to(bool)
        y = torch.autograd.functional.vhp(
            func, x, v, create_graph=create_graph, strict=strict
        )

        return y

    @autotest(n=5, check_graph=False)
    def test_vhp_tensor_cos_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = _exp_tensor_cos
        v = torch.ones_like(x)
        create_graph = random().to(bool)
        strict = random().to(bool)
        y = torch.autograd.functional.vhp(
            func, x, v, create_graph=create_graph, strict=strict
        )

        return y

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_autograd_functional_jacobian(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = torch.sin
        jac = torch.autograd.functional.jacobian(func, x, create_graph=True)
        return jac

    @autotest(n=5, auto_backward=False, check_graph=False)
    def test_autograd_functional_hessian(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = torch.sin
        hess = torch.autograd.functional.hessian(func, x, create_graph=True)
        return hess


if __name__ == "__main__":
    unittest.main()
