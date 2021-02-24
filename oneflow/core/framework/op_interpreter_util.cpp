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
#include "oneflow/core/framework/op_interpreter_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"

namespace oneflow {
namespace one {

typedef HashMap<std::string, std::shared_ptr<compatible_py::BlobObject>> Bn2BlobObjectMap;

/*static*/ std::shared_ptr<cfg::OpAttribute> OpInterpUtil::AddBuiltinOpAndInferOpAttribute(
    const OperatorConf& op_conf, const bool is_mirrored_strategy_enabled) {
  std::shared_ptr<OpAttribute> op_attribute = [&]() {
    auto infer_ctx = GetCurInferCtx().GetOrThrow();
    if (is_mirrored_strategy_enabled) {
      return infer_ctx->AddAndInferMirroredOp(op_conf).GetPtrOrThrow();
    } else {
      return infer_ctx->AddAndInferConsistentOp(op_conf).GetPtrOrThrow();
    }
  }();
  return std::make_shared<cfg::OpAttribute>(*op_attribute);
}

/*static*/ std::shared_ptr<cfg::OpAttribute> AddBuiltinOpAndInferOpAttribute(
    const BuiltinOpExpr* op_expr, const std::shared_ptr<Scope>& scope,
    const bool is_mirrored_strategy_enabled) {
  OperatorConf&& op_conf = OpInterpUtil::GenBuiltinOpConf(op_expr);
  int64_t symbol_id = scope->symbol_id().GetOrThrow();
  op_conf.set_scope_symbol_id(symbol_id);
  if (!op_conf.has_device_tag()) {
    op_conf.set_device_tag(scope->device_parallel_desc_symbol()->device_tag());
  }
  return OpInterpUtil::AddBuiltinOpAndInferOpAttribute(op_conf, is_mirrored_strategy_enabled);
}

/*static*/ std::string OpInterpUtil::GetJobNameScopePrefix(const std::shared_ptr<Session>& session,
                                                           const std::string& job_name) {
  // TODO
  return std::string("");
}

/*static*/ OperatorConf&& OpInterpUtil::GenBuiltinOpConf(const BuiltinOpExpr* op_expr) {
  OperatorConf op_conf;
  op_expr->BuildOpConf(&op_conf);
  return std::move(op_conf);
}

/*static*/ OperatorConf&& OpInterpUtil::GenModelInitOpConf(const OperatorConf& variable_conf) {
  OperatorConf model_init_op_conf;
  return std::move(model_init_op_conf);
}

/*static*/ OperatorConf&& OpInterpUtil::GenModelIOPathInputOpConf() {
  OperatorConf path_input_op_conf;
  return std::move(path_input_op_conf);
}

/*static*/ OperatorConf&& OpInterpUtil::GenModelLoadOpConf() {
  OperatorConf model_load_op_conf;
  return std::move(model_load_op_conf);
}

/*static*/ std::function<void(const std::shared_ptr<InstructionsBuilder>&)>
OpInterpUtil::BuildModelInitOrIOPathInputInstruction(
    const OperatorConf& op_conf, const std::shared_ptr<Bn2BlobObjectMap>& bn2blob_object) {
  using namespace std::placeholders;
  return [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    auto scope = GetCurrentScope().GetPtrOrThrow();
    auto op_attribute = OpInterpUtil::InferOpAttribute(op_conf, scope, Bn2BlobObjectMap{});
    auto parallel_conf =
        std::make_shared<cfg::ParallelConf>(scope->device_parallel_desc_symbol()->parallel_conf());
    const auto* boxing_util = Global<ForeignBoxingUtil>::Get();
    builder->StatelessCall(op_attribute, parallel_conf, bn2blob_object,
                           std::bind(&ForeignBoxingUtil::BoxingTo, boxing_util, _1, _2, _3));
  };
}

/*static*/ std::function<void(const std::shared_ptr<InstructionsBuilder>&)>
OpInterpUtil::BuildFeedPathInstruction(const std::shared_ptr<Bn2BlobObjectMap>& bn2blob_object) {
  // TODO
  int callback_id = -1;
  return [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    const auto& blob_object = bn2blob_object->at("out");
    builder->FeedBlob(blob_object, callback_id);
    builder->InsertRemoveForeignCallbackInstruction(blob_object->object_id(), callback_id);
  };
}

/*static*/ std::shared_ptr<compatible_py::BlobObject> OpInterpUtil::EagerRunModelInit(
    const OperatorConf& op_conf) {
  auto&& model_init_conf = GenModelInitOpConf(op_conf);
  std::shared_ptr<Bn2BlobObjectMap> bn2blob_object(new Bn2BlobObjectMap{});

  auto BuildModelInitInstruction =
      BuildModelInitOrIOPathInputInstruction(model_init_conf, bn2blob_object);
  LogicalRun(BuildModelInitInstruction).GetOrThrow();
  return bn2blob_object->at("out_0");
}

/*static*/ std::shared_ptr<compatible_py::BlobObject> OpInterpUtil::EagerRunModelLoad(
    const OperatorConf& op_conf, const std::string& snapshot_path) {
  using namespace std::placeholders;
  Path path(snapshot_path);
  CHECK(path.basename() == "out");
  CHECK(path.dirname() == op_conf.name());

  auto&& path_input_op_conf = GenModelIOPathInputOpConf();

  std::shared_ptr<Bn2BlobObjectMap> bn2blob_object(new Bn2BlobObjectMap{});
  auto BuildModelIOPathInputInstruction =
      BuildModelInitOrIOPathInputInstruction(path_input_op_conf, bn2blob_object);
  auto _BuildFeedPathInstruction = BuildFeedPathInstruction(bn2blob_object);

  std::shared_ptr<Bn2BlobObjectMap> model_load_blob_objects(new Bn2BlobObjectMap{});
  auto&& model_load_op_conf = GenModelLoadOpConf();
  auto BuildModelLoadInstruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    auto&& scope = GetCurrentScope().GetPtrOrThrow();
    const auto& blob_object = bn2blob_object->at("out");
    (*model_load_blob_objects)["path"] = blob_object;
    auto op_attribute =
        OpInterpUtil::InferOpAttribute(model_load_op_conf, scope, *model_load_blob_objects);
    auto parallel_conf =
        std::make_shared<cfg::ParallelConf>(scope->device_parallel_desc_symbol()->parallel_conf());
    const auto* boxing_util = Global<ForeignBoxingUtil>::Get();
    builder->StatelessCall(op_attribute, parallel_conf, model_load_blob_objects,
                           std::bind(&ForeignBoxingUtil::BoxingTo, boxing_util, _1, _2, _3));
  };

  LogicalRun(BuildModelIOPathInputInstruction).GetOrThrow();
  LogicalRun(_BuildFeedPathInstruction).GetOrThrow();
  LogicalRun(BuildModelLoadInstruction).GetOrThrow();
  return model_load_blob_objects->at("out_0");
}

/*static*/ void OpInterpUtil::Assign(
    const std::shared_ptr<compatible_py::BlobObject>& target_blob_object,
    const std::shared_ptr<compatible_py::BlobObject>& blob_object) {
  auto BuildAssignInstruction = [&](const std::shared_ptr<InstructionsBuilder>& builder) {
    const auto* boxing_util = Global<ForeignBoxingUtil>::Get();
    auto new_parallel_desc_symbol = boxing_util->TryReplaceDeviceTag(
        builder, target_blob_object->parallel_desc_symbol(), "cpu");
    auto consumer_op_arg_parallel_attr = std::make_shared<compatible_py::OpArgParallelAttribute>(
        new_parallel_desc_symbol, target_blob_object->op_arg_parallel_attr()->sbp_parallel(),
        target_blob_object->op_arg_parallel_attr()->opt_mirrored_parallel());
    auto tmp_blob_object =
        boxing_util->BoxingTo(builder, blob_object, consumer_op_arg_parallel_attr);
    boxing_util->Assign(builder, target_blob_object, tmp_blob_object);
  };
  LogicalRun(BuildAssignInstruction).GetOrThrow();
}

/*static*/ void OpInterpUtil::InitVariableOutputBlob(const std::shared_ptr<Session>& session,
                                                     const std::shared_ptr<Tensor>& output,
                                                     const OpAttribute& op_attribute) {
  const auto& op_conf = op_attribute.op_conf();
  const auto& snapshot_path = session->snapshot_mgr()->get_snapshot_path(op_conf.name());

  std::shared_ptr<compatible_py::BlobObject> temp_blob_object;
  if (snapshot_path.empty()) {
    temp_blob_object = OpInterpUtil::EagerRunModelInit(op_conf);
  } else {
    temp_blob_object = OpInterpUtil::EagerRunModelLoad(op_conf, snapshot_path);
  }
  auto target_blob_object = output->blob_object();
  OpInterpUtil::Assign(target_blob_object, temp_blob_object);
}

/*static*/ std::shared_ptr<cfg::OpAttribute> OpInterpUtil::InferOpAttribute(
    const OperatorConf& op_conf, const std::shared_ptr<Scope>& scope,
    const Bn2BlobObjectMap& ibn2blob_object) {
  auto& mutable_op_conf = const_cast<OperatorConf&>(op_conf);
  mutable_op_conf.set_scope_symbol_id(scope->symbol_id().GetOrThrow());
  OpNodeSignature upstream_signature;
  if (ibn2blob_object.size()) {
    std::shared_ptr<cfg::OpNodeSignature> cfg_upstream_signature(new cfg::OpNodeSignature);
    for (const auto& it : ibn2blob_object) {
      it.second->op_arg_parallel_attr()->DumpToOpNodeSignature(it.first, cfg_upstream_signature);
      it.second->op_arg_blob_attr()->DumpToOpNodeSignature(it.first, cfg_upstream_signature);
    }
    cfg_upstream_signature->ToProto(&upstream_signature);
  }
  const auto&& op =
      ConstructAndInferOp(mutable_op_conf, upstream_signature, *scope).GetPtrOrThrow();
  const auto& op_attribute = op->GetOpAttributeWithoutOpNameAndLbn();
  return std::make_shared<cfg::OpAttribute>(*op_attribute);
}

}  // namespace one
}  // namespace oneflow
