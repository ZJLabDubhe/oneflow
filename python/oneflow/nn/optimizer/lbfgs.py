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
import collections
import math
from typing import Callable, Dict, Iterator, List, Tuple, Union
from functools import reduce
from collections import defaultdict, abc as container_abcs
import numpy as np

import oneflow as flow
from oneflow.nn.optimizer.optimizer import Optimizer, ParamGroup
from oneflow.nn.parameter import Parameter
from torch._six import inf
import functools

"""
由于目前只是实现demo的状态，所以本文件不需要编译可以直接运行
理论上其实并不需要放在该文件夹下，因为是用绝对路径引用
然后目前的test代码也放在本文件末尾了，这是torch官方给的两个test中的一个
该test极为简单，没有进入lbfgs算法核心部分
手上有个小bug在调，等调完这个bug会引入另一段test代码
暂时没有按代码规范来写，里面很多注解也是中文，为了方便理解
后期等正式上线版本会进行代码规范调整
运行必要环境：oneflow，torch
"""


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # 三次线性插值法 用于寻找合适的步长 此处仅用于两点之间
    # 数值优化第76页
    # 大概作用是：优化算法主要是找方向和步长，线搜索主要用来寻找合适的
    # 步长，因为他可以利用之前的一些信息，
    # 原理大概类似于先给个左右范围，限定条件是wolfe，然后逐步缩小
    # 该算法已封装好没必要动
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1 ** 2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.


def _strong_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    # x(k+1) = x(k) + a(k) * p(k)
    # 优化算法核心公式如上所示，此处a是步长，p是方向
    # strong wolfe condition 总之就是加上了一堆限制条件很找步长
    d_norm = flow.max(flow.abs(d))
    g = g.clone().contiguous()
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = flow.dot(g_new,d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone().contiguous()]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if flow.abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone().contiguous()]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone().contiguous()
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = flow.dot(g_new,d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = flow.dot(g_new, d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone().contiguous()
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone().contiguous()
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


class Lbfgs(Optimizer):
    r"""Implements Adagrad Optimizer.
        The formula is:
        .. math::
            & S_{t} = S_{t-1} + grad \odot grad
            & decay\_lr = \frac{learning\_rate}{(1 + (train\_step - 1) * lr\_decay)}
            & X_{t} = X_{t-1} - \frac{decay\_lr}{\sqrt{S_{t} + \epsilon}} \odot grad
        Args:
            params (Union[Iterator[Parameter], List[Dict]]): iterable of parameters to optimize or dicts defining
            parameter groups
            lr (float, optional): The learning rate. Defaults to 0.001.
            lr_decay (float, optional): The decay factor of learning rate. Defaults to 0.0.
            weight_decay (float, optional): The weight decay. Defaults to 0.
            initial_accumulator_value (float, optional): The initial value of S. Defaults to 0.0.
            eps (float, optional): A small constant terms added to the denominator to improve numerical stability. Defaults to 1e-10.
        For example:
        Example 1:
        .. code-block:: python
            # Assume net is a custom model.
            lbfgs = flow.optim.Lbfgs(net.parameters(), lr=1e-3)
            for epoch in range(epochs):
                # Read data, Compute the loss and so on.
                # ...
                loss.backward()
                lbfgs.step()
                lbfgs.zero_grad()
        Example 2:
        .. code-block:: python
            # Assume net is a custom model.
            lbfgs = flow.optim.Lbfgs(
                [
                    {
                        "params": net.parameters(),
                        "lr": learning_rate,
                        "clip_grad_max_norm": 0.5,
                        "clip_grad_norm_type": 2.0,
                    }
                ],
            )
            for epoch in range(epochs):
                # Read data, Compute the loss and so on.
                # ...
                loss.backward()
                lbfgs.clip_grad()
                lbfgs.step()
                lbfgs.zero_grad()
        If you want to use clip_grad, you can refer this example.
        For more details of `clip_grad_max_norm` and `clip_grad_norm_type`, you can refer to :func:`oneflow.nn.utils.clip_grad_norm_`.
        """

    def __init__(
            self,
            params: Union[Iterator[Parameter], List[Dict]],
            lr: float = 0.001,
            max_iter=20,
            max_eval=None,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=100,
            line_search_fn=None,
    ):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        options = dict()
        options["lr"] = lr
        options["max_iter"] = max_iter
        options["max_eval"] = max_eval
        options["tolerance_grad"] = tolerance_grad
        options["tolerance_change"] = tolerance_change
        options["history_size"] = history_size
        options["line_search_fn"] = line_search_fn
        super().__init__(params, options)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._state = defaultdict(dict)
        #
        # print('params')
        # print(params)
        #
        # print('listpa')
        # print(list(params))
        #
        # print('paramgroups')

        # param_groups = list(params)

        # for param_group in self.param_groups:
        #     for param in param_group.parameters:
        #         print(param)

        # for param_group in self.param_groups:
        #     for param in param_group.parameters:
        #         assert param.is_leaf, "parameters must be leaf tensor"
        #         self._state[param] = dict()
        #
        # for param_group in self.param_groups:
        #     lr = param_group["lr"]
        #     print(lr)
        #     l2 = param_group["weight_decay"]
        #     print(l2)

        # if len(param_groups) == 0:
        #     raise ValueError("optimizer got an empty parameter list")
        # if not isinstance(param_groups[0], dict):
        #     param_groups = [{'params': param_groups}]
        #
        # for param_group in param_groups:
        #     self.add_param_group(param_group)
        #
        # print(self.param_groups)
        #
        group = self.param_groups[0]  # 这里依然是object

        self.parameters = group._parameters
        # print(self.parameters)
        # print('point')
        # print(params[0]['params'])
        # 这边较pytorch做了一点修改，我看pytorch原处这里是先把input的params存进param_group，再给赋值进self._params
        # 这边因为param_groups和oneflow有些区别，所以直接赋值进去了

        self._numel_cache = None

    def _numel(self):
        # numel 用于返回张量数量
        # reduce 用于列表内元素求和

        # 用来返回函数内张量总数
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self.parameters, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        # 求函数梯度（？）书上公式是这样
        # 应该没错，假如函数没梯度则返回等数量的0填充tensor
        views = []
        for p in self.parameters:
            if p.grad is None:

                # 这边zeros初始化的不合理，有点问题

                view = flow.tensor(np.zeros(p.numel()))
            # 下面有个if else给我去掉了，这个to_dense主要是考虑稀疏矩阵的
            # oneflow里好像没有对应操作就直接去掉了
            # elif p.grad.is_sparse:
            #     view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return flow.cat(views, 0)

    def _add_grad(self, step_size, update):
        # 梯度累加，没看具体逻辑
        offset = 0
        for p in self.parameters:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            # p.add_(update[offset:offset + numel].view(p.size()), alpha = step_size)
            p.add_((update[offset:offset + numel].view(p.size())) * step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        # torch里的clone和copy有本质区别，具体可以看
        # https://blog.csdn.net/qq_40438388/article/details/106860180
        # 这边每一个clone都做了contiguous操作，不知道目的
        return [p.clone().contiguous() for p in self.parameters]

    def _set_param(self, params_data):
        for p, pdata in zip(self.parameters, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        # LBFGS一个很关键的部分
        # 这是optimizer里面唯一一个用了闭包的，且必须用闭包的
        # 这部分好像是测试一下梯度下降的结果是否符合标准？看方法名差不多
        self._add_grad(t, d)
        loss = closure().float()
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    def step(self, closure: Callable = None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1
        with flow.no_grad():

            # 其实一直没搞懂 loss = closure()这部分的代码逻辑
            # loss到底从哪计算的
            # never mind，这部分torch和oneflow应该对齐的所以没必要太执着
            closure = flow.grad_enable()(closure)

            group = self.param_groups[0]  # 这里依然是object
            lr = group['lr']
            max_iter = group['max_iter']
            max_eval = group['max_eval']
            tolerance_grad = group['tolerance_grad']
            tolerance_change = group['tolerance_change']
            line_search_fn = group['line_search_fn']
            history_size = group['history_size']

            # NOTE: LBFGS has only global state, but we register it as state for
            # the first param, because this helps with casting in load_state_dict
            # 这部分state更新很头疼，非常头疼
            # 主要问题在于说torch.optimizer自带state更新机制
            # oneflow里面大概是没有的，目前对于optimizer里的代码研究还没有那么深入，有点菜鸡
            # 这部分目前在一一打印来试图debug
            # print('selfstate')
            # print(self._state)
            # print('params0')
            # print(self.parameters[0])
            # print('selfstateparams')
            # print(self._state[self.parameters[0]])
            state = self._state[self.parameters[0]]
            # 主要报错点，TypeError: 'generator' object is not subscriptable
            # 报错点2，添加了print，则_gather_flat_grad里的view_as会报错，不添加则能顺利运行

            state.setdefault('func_evals', 0)
            state.setdefault('n_iter', 0)

            # evaluate initial f(x) and df/dx
            orig_loss = closure()
            # print('orig_loss')
            # print(orig_loss)
            loss = orig_loss.float()
            current_evals = 1
            state['func_evals'] += 1

            flat_grad = self._gather_flat_grad()
            opt_cond = flow.max(flow.abs(flat_grad)) <= tolerance_grad

            # optimal condition
            # if opt_cond:
            #     return orig_loss

            # 同 state更新问题
            # 不过暂时代码还没有运行到这边
            d = state.get('d')
            t = state.get('t')
            old_dirs = state.get('old_dirs')
            old_stps = state.get('old_stps')
            ro = state.get('ro')
            H_diag = state.get('H_diag')
            prev_flat_grad = state.get('prev_flat_grad')
            prev_loss = state.get('prev_loss')

            n_iter = 0
            # optimize for a max of max_iter iterations
            while n_iter < max_iter:
                # keep track of nb of iterations
                n_iter += 1
                state['n_iter'] += 1
                ############################################################
                # compute gradient descent direction
                ############################################################
                if state['n_iter'] == 1:
                    d = flat_grad.neg()
                    old_dirs = []
                    old_stps = []
                    ro = []
                    H_diag = 1
                else:
                    # do lbfgs update (update memory)
                    y = flat_grad.sub(prev_flat_grad)
                    s = d.mul(t)
                    ys = flow.dot(y,s)  # y*s
                    if ys > 1e-10:
                        # updating memory
                        if len(old_dirs) == history_size:
                            # shift history by one (limited-memory)
                            old_dirs.pop(0)
                            old_stps.pop(0)
                            ro.pop(0)

                        # store new direction/step
                        old_dirs.append(y)
                        old_stps.append(s)
                        ro.append(1. / ys)

                        # update scale of initial Hessian approximation
                        H_diag = ys / (flow.dot(y,y))  # (y*y)

                    # compute the approximate (L-BFGS) inverse Hessian
                    # multiplied by the gradient
                    num_old = len(old_dirs)

                    if 'al' not in state:
                        state['al'] = [None] * history_size
                    al = state['al']

                    # iteration in L-BFGS loop collapsed to use just one buffer
                    # two loop recursion
                    # 数值优化第196页，主要是因为Hk难求，所以用其他的方式间接求Hk
                    # 主要方法是用two loop recursion
                    q = flat_grad.neg()
                    for i in range(num_old - 1, -1, -1):
                        al[i] = flow.dot(old_stps[i],q) * ro[i]
                        q = q + old_dirs[i] * (-al[i])

                    # multiply by initial Hessian
                    # r/d is the final direction
                    d = r = flow.mul(q, H_diag)
                    for i in range(num_old):
                        be_i = flow.dot(old_dirs[i], r) * ro[i]
                        r = r + old_stps[i] * (al[i] - be_i)

                if prev_flat_grad is None:
                    prev_flat_grad = flat_grad.clone().contiguous()
                else:
                    prev_flat_grad.copy_(flat_grad)
                prev_loss = loss

                ############################################################
                # compute step length
                ############################################################
                # reset initial guess for step size
                if state['n_iter'] == 1:
                    t = min(1., 1. / flow.sum(flow.abs(flat_grad))) * lr
                else:
                    t = lr

                # directional derivative
                gtd = flow.dot(flat_grad, d)  # g * d

                # directional derivative is below tolerance
                if gtd > -tolerance_change:
                    break
                # demo1 运行到此处

                # optional line search: user function
                ls_func_evals = 0
                if line_search_fn is not None:
                    # perform line search, using user function
                    if line_search_fn != "strong_wolfe":
                        raise RuntimeError("only 'strong_wolfe' is supported")
                    else:
                        x_init = self._clone_param()

                        def obj_func(x, t, d):
                            return self._directional_evaluate(closure, x, t, d)

                        loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                            obj_func, x_init, t, d, loss, flat_grad, gtd)
                    self._add_grad(t, d)
                    opt_cond = flow.max(flow.abs(flat_grad)) <= tolerance_grad
                else:
                    # no line search, simply move with fixed-step
                    self._add_grad(t, d)
                    if n_iter != max_iter:
                        # re-evaluate function only if not in last iteration
                        # the reason we do this: in a stochastic setting,
                        # no use to re-evaluate that function here
                        with flow.enable_grad():
                            loss = closure().float()
                        flat_grad = self._gather_flat_grad()
                        opt_cond = flow.max(flow.abs(flat_grad)) <= tolerance_grad
                        ls_func_evals = 1

                # update func eval
                current_evals += ls_func_evals
                state['func_evals'] += ls_func_evals

                ############################################################
                # check conditions
                ############################################################
                if n_iter == max_iter:
                    break

                if current_evals >= max_eval:
                    break

                # optimal condition
                if opt_cond:
                    break

                # lack of progress
                if flow.max(flow.abs(flow.mul(d,t))) <= tolerance_change:
                    break

                if flow.abs(loss - prev_loss) < tolerance_change:
                    break

            state['d'] = d
            state['t'] = t
            state['old_dirs'] = old_dirs
            state['old_stps'] = old_stps
            state['ro'] = ro
            state['H_diag'] = H_diag
            state['prev_flat_grad'] = prev_flat_grad
            state['prev_loss'] = prev_loss

            # self._state["step"] = self._state["step"] + 1

            return orig_loss

    def _generate_conf_for_graph(self, train_conf, vars_conf):
        # 这部分是oneflow不同于torch的一块
        # 姑且先从oneflow中的adagrad引进了这块
        # 代码暂时没有运行到这边

        new_opt_confs = []
        for param_group in self.param_groups:
            optimizer_conf = train_conf.optimizer_conf.add()

            lr = (
                param_group["initial_lr"]
                if "initial_lr" in param_group
                else param_group["lr"]
            )
            l2 = param_group["weight_decay"]
            initial_accumulator_value = param_group["initial_accumulator_value"]
            lr_decay = param_group["lr_decay"]
            epsilon = param_group["eps"]

            optimizer_conf.base_learning_rate = lr
            optimizer_conf.adagrad_conf.initial_accumulator_value = (
                initial_accumulator_value
            )
            optimizer_conf.adagrad_conf.lr_decay = lr_decay
            optimizer_conf.adagrad_conf.epsilon = epsilon

            self._generate_grad_clip_conf_for_optim_conf(param_group, optimizer_conf)

            for param in param_group.parameters:
                vars_conf[param].l2 = l2
                if param.requires_grad:
                    optimizer_conf.variable_op_names.append(vars_conf[param].name)

            new_opt_confs.append(optimizer_conf)
        return new_opt_confs

    # def add_param_group(self, param_group):
    #     r"""Add a param group to the :class:`Optimizer` s `param_groups`.
    #     This can be useful when fine tuning a pre-trained network as frozen layers can be made
    #     trainable and added to the :class:`Optimizer` as training progresses.
    #     Args:
    #         param_group (dict): Specifies what Tensors should be optimized along with group
    #         specific optimization options.
    #     """
    #
    #     # 这是torch里面一个更新param的方法
    #     # oneflow这块好像搞的不是太行，所以直接引用过来了
    #     assert isinstance(param_group, dict), "param group must be a dict"
    #
    #     params = param_group['params']
    #     if isinstance(params, flow.Tensor):
    #         param_group['params'] = [params]
    #     elif isinstance(params, set):
    #         raise TypeError('optimizer parameters need to be organized in ordered collections, but '
    #                         'the ordering of tensors in sets will change between runs. Please use a list instead.')
    #     else:
    #         param_group['params'] = list(params)
    #
    #     for param in param_group['params']:
    #         if not isinstance(param, flow.Tensor):
    #             raise TypeError("optimizer can only optimize Tensors, "
    #                             "but one of the params is " + param.type())
    #         if not param.is_leaf:
    #             raise ValueError("can't optimize a non-leaf Tensor")
    #
    #     for name, default in self.defaults.items():
    #         if default is required and name not in param_group:
    #             raise ValueError("parameter group didn't specify a value of required optimization parameter " +
    #                              name)
    #         else:
    #             param_group.setdefault(name, default)
    #
    #     params = param_group['params']
    #     if len(params) != len(set(params)):
    #         print("optimizer contains a parameter group with duplicate parameters; "
    #                       "in future, this will cause an error; "
    #                       "see github.com/pytorch/pytorch/issues/40967 for more information")
    #
    #     param_set = set()
    #     for group in self.param_groups:
    #         param_set.update(set(group['params']))
    #
    #     if not param_set.isdisjoint(set(param_group['params'])):
    #         raise ValueError("some parameters appear in more than one parameter group")
    #
    #     self.param_groups.append(param_group)


def b():
    params = [flow.randn(10, 5), flow.randn(10)]
    opt1 = Lbfgs(params, 0.01, tolerance_grad=inf)
    # opt2 = Lbfgs(params, 0.01, tolerance_grad=-inf)

    def closure():
        return flow.tensor([10])

    res1 = opt1.step(closure)
    # res2 = opt2.step(closure)
    print(res1)
    # print(res2)

b()
