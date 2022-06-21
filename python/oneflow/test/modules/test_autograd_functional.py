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


@flow.unittest.skip_unless_1n1d()
class TestAutogradFunctiona(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False, check_graph=False)
    def test_autograd_functional_jacobian(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = torch.sin
        jac = torch.autograd.functional.jacobian(func, x, create_graph=True)
        return jac

    @autotest(n=10, auto_backward=False, check_graph=False)
    def test_autograd_functional_hessian(test_case):
        device = random_device()
        ndim = random(1, 4).to(int)
        x = random_tensor(ndim=ndim, requires_grad=True).to(device)
        func = torch.sin
        hessian = torch.autograd.functional.hessian(func, x, create_graph=True)
        return hessian


if __name__ == "__main__":
    unittest.main()
