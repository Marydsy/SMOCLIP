from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from itertools import combinations
import random
from torch.cuda.amp import autocast  # 导入 autocast
from typing import Optional
try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
def sparse_contrastive_loss(logits, labels, k_sparse=10):
    # logits: (batch_size, batch_size), 对比学习的相似度矩阵
    pos_logits = logits[range(len(logits)), labels]  # 正样本logits（对角线）
    topk_neg_logits = torch.topk(logits, k=k_sparse, dim=1)[0]  # 每行取top-k负样本
    neg_loss = torch.logsumexp(topk_neg_logits, dim=1)  # 稀疏负样本损失
    loss = neg_loss - pos_logits
    return loss.mean()

# Step 1: Compute Shapley Values (Simplified using Monte Carlo Sampling)
def compute_shapley(logits, num_samples=100):
    """
    Approximate Shapley value for each player (image-text pair or feature).
    """
    device = logits.device  # 确保设备一致
    num_players = logits.shape[0]
    shapley_values = torch.zeros(num_players, device=device)  # 分配到 logits 的设备
    
    for player_idx in range(num_players):
        for _ in range(num_samples):
            # 随机生成一个子集
            subset = random.sample(range(num_players), k=random.randint(0, num_players - 1))
            subset = torch.tensor(subset, dtype=torch.long, device=device)  # 确保 subset 的设备和数据类型一致
            
            if player_idx not in subset:
                # 将当前玩家加入子集
                subset_with_player = torch.cat((subset, torch.tensor([player_idx], dtype=torch.long, device=device)))
                
                # 计算子集的值
                v_subset = logits[subset].sum()
                v_subset_with_player = logits[subset_with_player].sum()
                
                # 更新 Shapley 值
                shapley_values[player_idx] += (v_subset_with_player - v_subset) / num_samples
                
    return shapley_values


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features
# #focal_loss
# class ClipLoss(nn.Module):
#     def __init__(
#             self,
#             local_loss=False,
#             gather_with_grad=False,
#             cache_labels=False,
#             rank=0,
#             world_size=1,
#             use_horovod=False,
#             alpha=0.25,  # Alpha parameter for Focal Loss
#             gamma=5.0,   # Gamma parameter for Focal Loss
#     ):
#         super().__init__()
#         self.local_loss = local_loss
#         self.gather_with_grad = gather_with_grad
#         self.cache_labels = cache_labels
#         self.rank = rank
#         self.world_size = world_size
#         self.use_horovod = use_horovod
#         self.alpha = alpha
#         self.gamma = gamma

#         # cache state
#         self.prev_num_logits = 0
#         self.labels = {}

#     def get_ground_truth(self, device, num_logits) -> torch.Tensor:
#         if self.prev_num_logits != num_logits or device not in self.labels:
#             labels = torch.arange(num_logits, device=device, dtype=torch.long)
#             if self.world_size > 1 and self.local_loss:
#                 labels = labels + num_logits * self.rank
#             if self.cache_labels:
#                 self.labels[device] = labels
#                 self.prev_num_logits = num_logits
#         else:
#             labels = self.labels[device]
#         return labels

#     def get_logits(self, image_features, text_features, logit_scale):
#         if self.world_size > 1:
#             all_image_features, all_text_features = gather_features(
#                 image_features,
#                 text_features,
#                 local_loss=self.local_loss,
#                 gather_with_grad=self.gather_with_grad,
#                 rank=self.rank,
#                 world_size=self.world_size,
#                 use_horovod=self.use_horovod,
#             )

#             if self.local_loss:
#                 logits_per_image = logit_scale * image_features @ all_text_features.T
#                 logits_per_text = logit_scale * text_features @ all_image_features.T
#             else:
#                 logits_per_image = logit_scale * all_image_features @ all_text_features.T
#                 logits_per_text = logits_per_image.T
#         else:
#             logits_per_image = logit_scale * image_features @ text_features.T
#             logits_per_text = logit_scale * text_features @ image_features.T

#         return logits_per_image, logits_per_text

#     def focal_loss(self, logits, labels):
#         with autocast():
#             # 计算 softmax 概率
#             probs = F.softmax(logits, dim=-1)
            
#             # 获取真实类别的概率
#             p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            
#             # 计算 Focal Loss
#             focal_loss = -self.alpha * (1 - p_t).pow(self.gamma) * torch.log(p_t + 1e-10)  # 添加 epsilon 防止数值不稳定
        
#         return focal_loss.mean()  # 返回批次的平均损失

#     def forward(self, image_features, text_features, logit_scale, output_dict=False):
#         device = image_features.device
        
#         # 使用混合精度计算 logits
#         with autocast():
#             logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
#             labels = self.get_ground_truth(device, logits_per_image.shape[0])

#             # 计算 Focal Loss
#             loss_image = self.focal_loss(logits_per_image, labels)
#             loss_text = self.focal_loss(logits_per_text, labels)

#             # 平均损失
#             total_loss = (loss_image + loss_text) / 2

#         return {"contrastive_loss": total_loss} if output_dict else total_loss
#auto+shap
# class ClipLoss(nn.Module):
#     def __init__(
#             self,
#             local_loss=False,
#             gather_with_grad=False,
#             cache_labels=False,
#             rank=0,
#             world_size=1,
#             use_horovod=False,
#     ):
#         super().__init__()
#         self.local_loss = local_loss
#         self.gather_with_grad = gather_with_grad
#         self.cache_labels = cache_labels
#         self.rank = rank
#         self.world_size = world_size
#         self.use_horovod = use_horovod

#         # cache state
#         self.prev_num_logits = 0
#         self.labels = {}

#     def get_ground_truth(self, device, num_logits) -> torch.Tensor:
#         # calculated ground-truth and cache if enabled
#         if self.prev_num_logits != num_logits or device not in self.labels:
#             labels = torch.arange(num_logits, device=device, dtype=torch.long)
#             if self.world_size > 1 and self.local_loss:
#                 labels = labels + num_logits * self.rank
#             if self.cache_labels:
#                 self.labels[device] = labels
#                 self.prev_num_logits = num_logits
#         else:
#             labels = self.labels[device]
#         return labels  # 256

#     def get_logits(self, image_features, text_features, logit_scale):  # logit_scale标量，缩放相似度分数
#         if self.world_size > 1:
#             all_image_features, all_text_features = gather_features(
#                 image_features,  # 256*512
#                 text_features,  # 256*512
#                 local_loss=self.local_loss,
#                 gather_with_grad=self.gather_with_grad,
#                 rank=self.rank,
#                 world_size=self.world_size,
#                 use_horovod=self.use_horovod,
#             )

#             if self.local_loss:
#                 logits_per_image = logit_scale * image_features @ all_text_features.T  # all_text_features.T是转置
#                 logits_per_text = logit_scale * text_features @ all_image_features.T
#             else:
#                 logits_per_image = logit_scale * all_image_features @ all_text_features.T  # 每个图像特征与每个文本特征的相似度分数
#                 logits_per_text = logits_per_image.T
#         else:
#             logits_per_image = logit_scale * image_features @ text_features.T  # text_features [256,512]
#             logits_per_text = logit_scale * text_features @ image_features.T  # @是矩阵乘法,得到相似度矩阵 [256,256]

#         return logits_per_image, logits_per_text  # 256*256 概率分布矩阵, logits_per_image[0]为图像 0 的预测分布，即和0到255文本的相似度

#     def forward(self, image_features, text_features, logit_scale, output_dict=False):
#         device = image_features.device

#         # 使用 autocast 包装前向计算
#         with autocast():
#             logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

#             labels = self.get_ground_truth(device, logits_per_image.shape[0])  # 256 [0,1,2,3,...,255],图像 0 应该匹配文本 0

#             # 计算 Shapley 值
#             shapley_image = compute_shapley(logits_per_image)
#             shapley_text = compute_shapley(logits_per_text)

#             # 归一化 Shapley 值
#             shapley_image = shapley_image / shapley_image.sum()
#             shapley_text = shapley_text / shapley_text.sum()

#             # 加权损失
#             total_loss = 0.5 * (
#                 torch.dot(shapley_image, F.cross_entropy(logits_per_image, labels, reduction='none')) +
#                 torch.dot(shapley_text, F.cross_entropy(logits_per_text, labels, reduction='none'))
#             )

#         return {"contrastive_loss": total_loss} if output_dict else total_loss
#原代码和shap
class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels #256

    def get_logits(self, image_features, text_features, logit_scale):#logit_scale标量，缩放相似度分数
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,#256*512
                text_features,#256*512
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T #all_text_features.T是转置
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T#每个图像特征与每个文本特征的相似度分数
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T #text_features [256,512]
            logits_per_text = logit_scale * text_features @ image_features.T #@是矩阵乘法,得到相似度矩阵 [256,256]
        
        return logits_per_image, logits_per_text#256*256 概率分布矩阵, logits_per_image[0]为图像 0 的预测分布，即和0到255文本的相似度

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])#256 [0,1,2,3,...,255],图像 0 应该匹配文本 0

        # negclip+smoothing
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text,labels)
            ) / 2
      

        return {"contrastive_loss": total_loss} if output_dict else total_loss

class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)#交叉熵

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)

class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    #加入了混合精度计算
    def __init__(
            self,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        # 使用 autocast 包装计算部分
        with autocast():
            logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
            labels = self.get_ground_truth(
                image_features.device,
                image_features.dtype,
                image_features.shape[0],
                negative_only=negative_only,
            )
            loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        # 使用 autocast 包装整个前向计算
        with autocast():
            loss = self._loss(image_features, text_features, logit_scale, logit_bias)

            if self.world_size > 1:
                if self.dist_impl == 'bidir':
                    right_rank = (self.rank + 1) % self.world_size
                    left_rank = (self.rank - 1 + self.world_size) % self.world_size
                    text_features_to_right = text_features_to_left = text_features
                    num_bidir, remainder = divmod(self.world_size - 1, 2)
                    for i in range(num_bidir):
                        text_features_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            text_features_to_left,
                            text_features_to_right,
                        )
                        for f in text_features_recv:
                            loss += self._loss(
                                image_features,
                                f,
                                logit_scale,
                                logit_bias,
                                negative_only=True,
                            )
                        text_features_to_left, text_features_to_right = text_features_recv

                    if remainder:
                        text_features_recv = neighbour_exchange_with_grad(
                            left_rank,
                            right_rank,
                            text_features_to_right
                        )
                        loss += self._loss(
                            image_features,
                            text_features_recv,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                elif self.dist_impl == "shift":
                    right_rank = (self.rank + 1) % self.world_size
                    left_rank = (self.rank - 1 + self.world_size) % self.world_size
                    text_features_to_right = text_features
                    for i in range(self.world_size - 1):
                        text_features_from_left = neighbour_exchange_with_grad(
                            left_rank,
                            right_rank,
                            text_features_to_right,
                        )
                        loss += self._loss(
                            image_features,
                            text_features_from_left,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                        text_features_to_right = text_features_from_left
                elif self.dist_impl == "reduce":
                    for i in range(self.world_size):
                        text_from_other = torch.distributed.nn.all_reduce(
                            text_features * (self.rank == i),
                            torch.distributed.ReduceOp.SUM,
                        )
                        loss += float(i != self.rank) * self._loss(
                            image_features,
                            text_from_other,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                elif self.dist_impl == "gather":
                    all_text = torch.distributed.nn.all_gather(text_features)
                    for i in range(self.world_size):
                        loss += float(i != self.rank) * self._loss(
                            image_features,
                            all_text[i],
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                else:
                    assert False

        return {"contrastive_loss": loss} if output_dict else loss
