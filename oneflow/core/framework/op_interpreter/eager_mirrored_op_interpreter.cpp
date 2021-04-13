/*
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
*/
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/memory/memory_case_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/user/kernels/stateful_opkernel.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace one {

static Maybe<void> NaiveInterpret(const BuiltinOpExpr& op_expr, const TensorTuple& inputs,
                                  TensorTuple* outputs) {
  // printf("mirrored NaiveInterpret\n");
  OF_PROFILER_RANGE_PUSH("NaiveInterpret");
  OF_PROFILER_RANGE_PUSH("NaiveInterpret: GetCurrentScope");
  std::shared_ptr<const ParallelDesc> parallel_desc;
  std::shared_ptr<const Device> device;
  const auto& scope = JUST(GetCurrentScope());
  OF_PROFILER_RANGE_POP();  // NaiveInterpret: GetCurrentScope

  OF_PROFILER_RANGE_PUSH("NaiveInterpret: device_parallel_desc_symbol");
  parallel_desc = scope->device_parallel_desc_symbol();
  OF_PROFILER_RANGE_POP();

  OF_PROFILER_RANGE_PUSH("NaiveInterpret: MakeDeviceByParallelDesc");
  device = JUST(Device::MakeDeviceByParallelDesc(*parallel_desc));
  OF_PROFILER_RANGE_POP();

  OF_PROFILER_RANGE_PUSH("NaiveInterpret: DeviceId4ParallelId");
  int64_t device_id = JUST(parallel_desc->DeviceId4ParallelId(0));
  OF_PROFILER_RANGE_POP();

  // TODO: async
  TensorsPtr eager_blob_objects =
      std::make_shared<std::vector<std::shared_ptr<eager::EagerBlobObject>>>();
  auto build_instruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    OF_PROFILER_RANGE_PUSH("build_instruction: make_shared");
    auto& user_op_expr = dynamic_cast<const UserOpExpr&>(op_expr);
    OF_PROFILER_RANGE_POP();
    // TODO:
    OF_PROFILER_RANGE_PUSH("build_instruction: set_device");
    user_op_expr.mut_kernel()->set_device(parallel_desc->device_type(), device_id,
                                          parallel_desc->device_tag());
    OF_PROFILER_RANGE_POP();

    OF_PROFILER_RANGE_PUSH("build_instruction: LocalCallOpKernel");
    builder->LocalCallOpKernel(user_op_expr.mut_kernel(), inputs, *outputs, eager_blob_objects,
                               parallel_desc);
    OF_PROFILER_RANGE_POP();
  };

  OF_PROFILER_RANGE_PUSH("build_instruction: PhysicalRun");
  JUST(PhysicalRun(build_instruction));
  OF_PROFILER_RANGE_POP();

  OF_PROFILER_RANGE_PUSH("build_instruction: BuildEagerMirroredTensorFromEagerBlobObject");
  for (int i = 0; i < outputs->size(); ++i) {
    (*outputs)[i] = CHECK_JUST(OpInterpUtil::BuildEagerMirroredTensorFromEagerBlobObject(
        (*eager_blob_objects)[i], device));
  }
  OF_PROFILER_RANGE_POP();

  OF_PROFILER_RANGE_POP();
  return Maybe<void>::Ok();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const UserOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  return NaiveInterpret(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const VariableOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  CHECK_EQ_OR_RETURN(inputs.size(), 0);
  CHECK_EQ_OR_RETURN(outputs->size(), 1);
  return NaiveInterpret(op_expr, inputs, outputs);
}

static Maybe<void> BuildAndRunMirroredCastInstruction(const BuiltinOpExpr& op_expr,
                                                      const TensorTuple& inputs,
                                                      TensorTuple* outputs) {
  // TODO()
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const CastToMirroredOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  return BuildAndRunMirroredCastInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const CastFromMirroredOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  return BuildAndRunMirroredCastInstruction(op_expr, inputs, outputs);
}

static Maybe<void> BuildAndRunDistributeSplitOrCloneInstruction(const BuiltinOpExpr& op_expr,
                                                                const TensorTuple& inputs,
                                                                TensorTuple* outputs) {
  // TODO()
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeSplitOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  return BuildAndRunDistributeSplitOrCloneInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeCloneOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  return BuildAndRunDistributeSplitOrCloneInstruction(op_expr, inputs, outputs);
}

static Maybe<void> BuildAndRunDistributeConcatAndAddInstruction(const BuiltinOpExpr& op_expr,
                                                                const TensorTuple& inputs,
                                                                TensorTuple* outputs) {
  // TODO()
  OF_UNIMPLEMENTED();
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeConcatOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  return BuildAndRunDistributeConcatAndAddInstruction(op_expr, inputs, outputs);
}

Maybe<void> EagerMirroredInterpreter::ApplyImpl(const DistributeAddOpExpr& op_expr,
                                                const TensorTuple& inputs,
                                                TensorTuple* outputs) const {
  return BuildAndRunDistributeConcatAndAddInstruction(op_expr, inputs, outputs);
}

}  // namespace one
}  // namespace oneflow
