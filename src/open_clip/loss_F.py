import torch
import torch.nn as nn
from torch.nn import functional as F

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
class ClipLoss_F(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            is_siglip=False
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.is_siglip = is_siglip
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        # num_logits equals to batchsize
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.is_siglip:
            logit_scale, logit_bias = logit_scale
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                if self.is_siglip:
                    # siglip 
                    logits_per_image = torch.matmul(image_features, all_text_features.t()) * logit_scale +logit_bias
                    logits_per_text = torch.matmul(text_features, all_image_features.t()) * logit_scale +logit_bias
                
                else:
                    logits_per_image = logit_scale * image_features @ all_text_features.T
                    logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                if self.is_siglip:
                    # siglip 
                    logits_per_image = torch.matmul(all_image_features, all_text_features.t()) * logit_scale + logit_bias
                    logits_per_text = logits_per_image.T
                else:
                    logits_per_image = logit_scale * all_image_features @ all_text_features.T
                    logits_per_text = logits_per_image.T
        else:
            if self.is_siglip:
                # siglip 
                logits_per_image = torch.matmul(image_features, text_features.t()) * logit_scale + logit_bias
                logits_per_text = torch.matmul(text_features, image_features.t()) * logit_scale + logit_bias
            else:
                logits_per_image = logit_scale * image_features @ text_features.T
                logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        if self.is_siglip:
            n = logits_per_image.shape[0]
            labels = 2 * torch.eye(n, device=device) - torch.ones(n, device = device) 
            total_loss = -torch.mean(F.logsigmoid(labels * logits_per_image))
        else:    
            labels = self.get_ground_truth(device, logits_per_image.shape[0])

            total_loss = (
                F.cross_entropy(logits_per_image, labels,label_smoothing=0.1) +
                F.cross_entropy(logits_per_text, labels,label_smoothing=0.1)
            ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss



class DCL(object):
    """
    Decoupled Contrastive Loss (DCL) proposed in https://arxiv.org/pdf/2110.06848.pdf
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        self.temperature = temperature
        self.weight_fn = weight_fn

    def __call__(self, z1, z2):
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * -1e10, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()


class ClipLoss_DCLF(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        is_siglip=False,
        use_dcl=True,  # Add flag for DCL usage
        dcl_temperature=0.1,  # Default temperature for DCL
        weight_fn=None,  # Optional weight function for DCL
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.is_siglip = is_siglip
        self.use_dcl = use_dcl  # Store the flag for DCL usage
        self.dcl = DCL(temperature=dcl_temperature, weight_fn=weight_fn) if use_dcl else None  # Initialize DCL if needed
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.is_siglip:
            logit_scale, logit_bias = logit_scale
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                if self.is_siglip:
                    logits_per_image = torch.matmul(image_features, all_text_features.t()) * logit_scale + logit_bias
                    logits_per_text = torch.matmul(text_features, all_image_features.t()) * logit_scale + logit_bias
                else:
                    logits_per_image = logit_scale * image_features @ all_text_features.T
                    logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                if self.is_siglip:
                    logits_per_image = torch.matmul(all_image_features, all_text_features.t()) * logit_scale + logit_bias
                    logits_per_text = logits_per_image.T
                else:
                    logits_per_image = logit_scale * all_image_features @ all_text_features.T
                    logits_per_text = logits_per_image.T
        else:
            if self.is_siglip:
                logits_per_image = torch.matmul(image_features, text_features.t()) * logit_scale + logit_bias
                logits_per_text = torch.matmul(text_features, image_features.t()) * logit_scale + logit_bias
            else:
                logits_per_image = logit_scale * image_features @ text_features.T
                logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        # Use DCL if the flag is set
        if self.use_dcl:
            dcl_loss = self.dcl(image_features, text_features)
            total_loss = dcl_loss
        else:
            if self.is_siglip:
                n = logits_per_image.shape[0]
                labels = 2 * torch.eye(n, device=device) - torch.ones(n, device=device)
                total_loss = -torch.mean(F.logsigmoid(labels * logits_per_image))
            else:
                labels = self.get_ground_truth(device, logits_per_image.shape[0])
                total_loss = (
                    F.cross_entropy(logits_per_image, labels, label_smoothing=0.1) +
                    F.cross_entropy(logits_per_text, labels, label_smoothing=0.1)
                ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


def gather_features_da(
        image_features,
        text_features,
        valid_caption_mask,
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
            all_valid_caption_mask=torch.cat(torch.distributed.nn.all_gather(valid_caption_mask), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            gathered_valid_caption_mask = [torch.zeros_like(valid_caption_mask) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            dist.all_gather(gathered_valid_caption_mask, valid_caption_mask)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                gathered_valid_caption_mask[rank] = valid_caption_mask
                
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
            all_valid_caption_mask = torch.cat(gathered_valid_caption_mask, dim=0)

    return all_image_features, all_text_features, all_valid_caption_mask

class Clip_DALoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            cmr_loss=False,
            imc_loss=False,
            hardnegative=False,
            imc_loss_weight=0.2,
            cmr_loss_weight=0.2,
            threshold_type='mean',
          
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
        self.cmr_loss=cmr_loss
        self.imc_loss=imc_loss
        self.imc_loss_weight=imc_loss_weight
        self.cmr_loss_weight=cmr_loss_weight
        self.threshold_type=threshold_type
        self.hardnegative=hardnegative
        
    def forward(self, image_features, text_features,valid_caption_mask, logit_scale, thresholds):
        """
        cross-modal ranking loss doesn't support local_loss and use_horovod 

        Different Losses:
            - hard negative: standard clip contrastive loss, assuming hard-negatives as extra negative for computing logits_per_image, logits_per_text is the same as clip
            - imc_loss: standard clip contrastive loss + contrastive loss on text embeddings (between ground truth caption embedding and hard-negative caption embedding)
            - cmr_loss: standard clip contrastive loss + rank loss between gt pair and hg pair
        """
        if isinstance(logit_scale, tuple):
            raise NotImplementedError("siglip not supported")
        device = image_features.device
        cmr_loss,imc_loss=0.0,0.0
        if self.world_size > 1:
            all_image_features, all_text_features, all_valid_caption_mask = gather_features_da(
                image_features, text_features, valid_caption_mask,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            
            caption_types=torch.tensor(([1]*image_features.shape[0]+[2]*image_features.shape[0]*4)*self.world_size)
            gt_all_text_features=all_text_features[caption_types==1] # batch_size * word_size
            da_all_text_features=all_text_features[caption_types==2] # 4 * batch_size * word_size
            gt_len,feature_size=all_image_features.shape[0],all_image_features.shape[-1]


            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                #extra hard negative loss
                if self.hardnegative:
                    all_text_features=torch.cat([gt_all_text_features,da_all_text_features])
                    logits_per_image = logit_scale * all_image_features @ all_text_features.T # batch_size * 5xbatch_size       
                else:
                    logits_per_image = logit_scale * all_image_features @ gt_all_text_features.T

                logits_per_text = logit_scale * gt_all_text_features @ all_image_features.T

                # cross-modal rank loss
                if self.cmr_loss:
                    da_logits_per_image= logit_scale * (da_all_text_features.reshape(gt_len,-1,feature_size)@ all_image_features.unsqueeze(-1)).squeeze() * all_valid_caption_mask
                    cmr_loss,thresholds=self.get_cmr_loss(logits_per_image,da_logits_per_image,all_valid_caption_mask,thresholds)
                
                # intra-modal contrastive loss
                if self.imc_loss:
                    text_embedding_matrix=logit_scale * gt_all_text_features @ da_all_text_features.T  #(all_batch_size,4*all_batch_size)
                    imc_loss+=self.get_imc_loss(logits_per_image,text_embedding_matrix)

        else:
        # not updating very long time
            gt_len,feature_size=image_features.shape[0],image_features.shape[-1]
            gt_text_features=text_features[:image_features.shape[0]]
            da_text_features=text_features[image_features.shape[0]:]
            all_text_features=torch.cat([gt_text_features,da_text_features])
            if self.hardnegative:
                logits_per_image = logit_scale * image_features @ all_text_features.T
            else:
                logits_per_image = logit_scale * image_features @ gt_text_features.T
            logits_per_text = logit_scale * gt_text_features @ image_features.T
            if self.cmr_loss:
                da_logits_per_image=  logit_scale * (da_text_features.reshape(gt_len,-1,feature_size)@ image_features.unsqueeze(-1)).squeeze() * valid_caption_mask
                cmr_loss,thresholds=self.get_cmr_loss(logits_per_image,da_logits_per_image,valid_caption_mask,thresholds)
            if self.imc_loss:
                text_embedding_matrix=logit_scale * gt_text_features @ da_text_features.T #(batch_size,4*batch_size)
                imc_loss=self.get_imc_loss(logits_per_image,text_embedding_matrix)         
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        if self.cmr_loss:
            total_loss+=cmr_loss*self.cmr_loss_weight
        if self.imc_loss:
            total_loss+=imc_loss*self.imc_loss_weight
            
        return total_loss,thresholds,cmr_loss,imc_loss
   
        
    def get_cmr_loss(self,gt_logits_per_image:torch.Tensor,da_logits_per_image:torch.Tensor,valid_caption_mask,thresholds:torch.Tensor) -> torch.Tensor:
        # calculating cmr loss
        gt_similarity=gt_logits_per_image.diag().reshape(-1,1).expand(da_logits_per_image.shape)
        # gt_similarity=gt_logits_per_image.gather(0,torch.arange(min(gt_logits_per_image.shape),device=gt_logits_per_image.device).reshape(1,-1)).reshape(min(gt_logits_per_image.shape),1).expand(da_logits_per_image.shape)
        cmr_loss=nn.functional.relu((thresholds+da_logits_per_image-gt_similarity))*valid_caption_mask


        # updating thresholds
        if self.threshold_type=='mean':
            mask = da_logits_per_image!=0
            average_similarity_for_types = (da_logits_per_image*mask).sum(dim=0)/mask.sum(dim=0)
            thresholds=(gt_similarity.mean(0)-average_similarity_for_types).expand(gt_similarity.shape)
            thresholds=thresholds.detach()
        elif self.threshold_type=='max':
            thresholds,max_indices=(gt_similarity*valid_caption_mask-da_logits_per_image).max(0)
            thresholds=thresholds.expand(gt_similarity.shape)/5
            thresholds=thresholds.detach()
        return cmr_loss.mean(),thresholds

    def get_imc_loss(self,gt_logits_per_image:torch.Tensor,embedding_matrix:torch.Tensor):
        """
        gt_logits_per_image: standard clip similarity matrix, diag is true gt similarity value : shape [batch_size,5xbatch_size]
        embedding_matrix: extra similarity matrix served as denominator in clip loss
        """
        
        logtis_matrix = embedding_matrix
        labels=torch.zeros(logtis_matrix.shape[0],device=logtis_matrix.device,dtype=torch.long)
        imc_loss=F.cross_entropy(logtis_matrix,labels)
        return imc_loss
        