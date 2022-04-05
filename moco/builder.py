# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoCCL(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, extra_sample_K=256, m=0.999, teacher_m=0.996, T=0.07, mlp=False):
        super(MoCoCCL, self).__init__()

        self.K = K
        self.extra_sample_K = extra_sample_K
        self.m = m
        self.T = T
        self.t = 0
        self.teacher_m = teacher_m
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        self.teacher = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
            self.teacher.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.teacher.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False 

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("extra_sample_queue", torch.randn(dim, self.extra_sample_K))
        self.extra_sample_queue = nn.functional.normalize(self.extra_sample_queue, dim=0)

        self.register_buffer("extra_sample_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_teacher(self):
        for param_q, param_t in zip(self.encoder_q.parameters(), self.teacher.parameters()):
            param_t.data = param_t.data * self.teacher_m + param_q.data * (1 - self.teacher_m)
            param_t.requires_grad = False

    @torch.no_grad()
    def reset_teacher(self):
        for param_q, param_t in zip(self.encoder_q.parameters(), self.teacher.parameters()):
            param_t.data.copy_(param_q.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def reset_k(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            # param_t.requires_grad = False

    def begin_incremental(self):
        self.extra_sample_queue[:, :] = self.queue[:, :self.extra_sample_K]

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, is_from_old):
        # update queue
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

        # update extra sample queue
        is_from_old = concat_all_gather(is_from_old)
        is_from_old = is_from_old.squeeze()
        idx = is_from_old==1
        if self.t>0 and idx.sum()>0:
            keys = keys[idx, :]
            bs = keys.shape[0]
            p1 = int(self.extra_sample_queue_ptr)
            if bs>=self.extra_sample_K:
                self.extra_sample_queue[:, :] = keys[bs-self.extra_sample_K:, :].t()
                self.extra_sample_queue_ptr[0] = 0
            else:
                carry = (p1+bs)//self.extra_sample_K
                remain = (p1+bs)%self.extra_sample_K
                if carry:
                    self.extra_sample_queue[:, p1:] = keys[:self.extra_sample_K-p1, :].t()
                    if remain:
                        self.extra_sample_queue[:, :remain] = keys[self.extra_sample_K-p1:, :].t()
                    self.extra_sample_queue_ptr[0] = remain
                else:
                    if remain:
                        self.extra_sample_queue[:, p1:remain] = keys.t()
                    self.extra_sample_queue_ptr[0] = remain

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_encoder_q(self, images):
        q = self.encoder_q(images)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        return q

    def forward(self, im_q, im_k=None, im_raw=None, is_from_old=None, mode='train', loss_fun=None, t=0):
        assert mode in ['train', 'feature']
        if mode == 'feature':
            return self.forward_encoder_q(im_q)
        else:
            return self.forward_train(im_q, im_k, im_raw, is_from_old, loss_fun, t)

    def forward_train(self, im_q, im_k, im_raw, is_from_old, criterion, t):
        # set up for incremental learning
        if self.t<t:
            if self.t==0:
                self.begin_incremental()
            self.t = t
            self.reset_teacher()
            self.reset_k()
            self.teacher.eval()
        
        ###### Original MoCo ######

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()


        ###### incremental version of MoCo ######
        incremental_loss = 0.0

        if self.t>0:

            # extra sample queue loss
            l_neg_extra_sample = torch.einsum('nc,ck->nk', [q, self.extra_sample_queue.clone().detach()])
            logits_extra_sample = torch.cat([l_pos, l_neg_extra_sample], dim=1)
            logits_extra_sample /= self.T
            labels_extra_sample = torch.zeros(logits_extra_sample.shape[0], dtype=torch.long).cuda()
            
            incremental_loss += 0.1 * criterion(logits_extra_sample, labels_extra_sample)

            # self-supervised knowledge distillation loss
            idx = is_from_old.squeeze()==1
            if idx.sum()>0:
                s_anchor = self.encoder_q(im_raw)
                s_anchor = nn.functional.normalize(s_anchor, dim=1)
                t_q, t_anchor = self.teacher(im_q), self.teacher(im_raw)
                t_q = nn.functional.normalize(t_q, dim=1)
                t_anchor = nn.functional.normalize(t_anchor, dim=1)

                s_q, s_anchor = q[idx,:], s_anchor[idx,:]
                t_q, t_anchor = t_q[idx,:].detach(), t_anchor[idx,:].detach()
                # s_q, s_anchor = q, s_anchor
                # t_q, t_anchor = t_q.detach(), t_anchor.detach()
            
                s_simi = torch.mm(s_q, s_anchor.t())
                t_simi = torch.mm(t_q, t_anchor.t())

                log_s_simi = F.log_softmax(s_simi / 0.07, dim=1)
                simi_knowledge = F.softmax(t_simi / 0.04, dim=1)

                kl_loss = F.kl_div(log_s_simi, simi_knowledge, \
                                reduction='batchmean')

                incremental_loss += 0.1 * kl_loss

        # update queue and extra sample queue
        self._dequeue_and_enqueue(k, is_from_old)

        return incremental_loss, logits, labels



class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output