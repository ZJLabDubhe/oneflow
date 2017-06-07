#include "oneflow/core/actor/model_update_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void MdUpdtCompActor::Init(const TaskProto& task_proto) {
  CompActor::Init(task_proto);
}

void MdUpdtCompActor::ProcessMsg(const ActorMsg& actor_msg,
                                 const ThreadContext& thread_ctx) {
  KernelContext kernel_ctx;
  kernel_ctx.cuda_stream = thread_ctx.compute_cuda_stream;
  if (actor_msg.msg_type() == ActorMsgType::kCmdMsg) {
    ProcessCommand(actor_msg.actor_cmd(), kernel_ctx);
  } else if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    TODO();
  } else {
    UNEXPECTED_RUN();
  }
}

void MdUpdtCompActor::ProcessCommand(ActorCmd cmd,
                                     const KernelContext& kernel_ctx) {
  if (cmd == ActorCmd::kInitModel) {
    ProcessInitModelCmd(kernel_ctx);
  } else {
    UNEXPECTED_RUN();
  }
}

void MdUpdtCompActor::ProcessInitModelCmd(const KernelContext& kernel_ctx) {
  Regst* model_regst = GetCurWriteableRegst("model");
  model_regst->set_model_version_id(0);
  Regst* model_tmp_regst = GetCurWriteableRegst("model_tmp");
  HashSet<const Kernel*> kernels;
  auto CollectKernelsFromLbn = [&kernels](const std::string& lbn) {
    std::string op_name = GetOpNameFromLbn(lbn);
    kernels.insert(KernelMgr::Singleton().GetKernelFromOpName(op_name));
  };
  model_regst->ForEachLbn(CollectKernelsFromLbn);
  model_tmp_regst->ForEachLbn(CollectKernelsFromLbn);
  for (const Kernel* kernel : kernels) {
    kernel->InitModelAndModelTmpBlobs(kernel_ctx, [&](const std::string& bn_in_op) {
      const std::string& lbn = kernel->Lbn4BnInOp(bn_in_op);
      Blob* ret = model_regst->GetBlobPtrFromLbn(lbn);
      if (ret == nullptr) { ret = model_tmp_regst->GetBlobPtrFromLbn(lbn); }
      CHECK(ret != nullptr);
      return ret;
    });
  }
}

REGISTER_ACTOR(kMdUpdtCompTask, true, MdUpdtCompActor);

}  // namespace oneflow
