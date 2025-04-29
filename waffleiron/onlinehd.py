import copy
from torchhd import functional
from torchhd import embeddings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .network.acdnet import GetACDNetModel

# Feature dimention dictionary for non-image datasets
feat_dim_dict = {
    'har': 561,
    'har_timeseries': 9,
    'isolet': 617,
    'mhealth': 21,
    'esc50': 30225
}

model_dim_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'acdnet': 50,
    'mobilenet_v2': 1280,
    'mobilenet_v3_large': 960,
    'mobilenet_v3_small': 576
}

class Model(nn.Module):
    def __init__(self, opt, num_classes, device):
        super().__init__()

        self.device = device

        # Record the current number of class hypervectors
        self.opt = opt
        self.num_classes = num_classes      # Used in supervised HD
        self.max_classes = opt.max_classes  # Used in unsupervised HD
        self.cur_classes = num_classes      # Used in semi unsupervised HD
        self.dataset = opt.dataset
        self.method = opt.method
        self.hd_dim = opt.dim
        self.temperature = opt.temperature
        self.win_size = opt.win_size

        self.flatten = torch.nn.Flatten()

        # set the input dimension
        self.feature_extractor = opt.feature_ext
        self.pretrained_on = opt.pretrained_on
        if self.dataset == 'mnist':
            self.input_dim = opt.size * opt.size
        elif self.dataset in ['har', 'isolet', 'mhealth', 
                              'har_timeseries', 'esc50']:
            self.input_dim = feat_dim_dict[opt.dataset]

        if self.feature_extractor != 'none':
            self.net = resnet50()
            self.net.classifier = nn.Identity()  # Disable the classification layer to only take features
            self.net.eval()
            self.input_dim = model_dim_dict[self.feature_extractor]

        self.hd_encoder = opt.hd_encoder
        if self.hd_encoder == 'rp':  # Random projection encoding
            # Generate a random projection matrix
            self.projection = embeddings.Projection(self.input_dim, self.hd_dim)

        elif self.hd_encoder == 'idlevel':  # ID-level encoding
            # Generate id-level value hv for each floating value
            self.value = embeddings.Level(opt.num_levels, self.hd_dim, 
                                          randomness=opt.randomness)
            # Create a random hv for each position, for binding with the value hv
            self.position = embeddings.Random(self.input_dim, self.hd_dim)

        elif self.hd_encoder == 'nonlinear':  # Nonlinear encoding
            self.nonlinear_projection = embeddings.Sinusoid(self.input_dim, self.hd_dim)
        
        else:  # No encoder, use raw samples
            if self.dataset == 'mhealth' or \
                self.dataset == 'har_timeseries':
                self.hd_dim = self.input_dim * self.win_size
            else:
                self.hd_dim = self.input_dim

        # Set classify
        if self.method == 'LifeHD':
            self.classify = nn.Linear(self.hd_dim, self.max_classes, bias=False)
            self.classify_sample_cnt = torch.zeros(self.max_classes).to(self.device)
            self.dist_mean = torch.zeros(self.max_classes).to(self.device)
            self.dist_std = torch.zeros(self.max_classes).to(self.device)
            self.last_edit = - np.ones(self.max_classes)

        elif self.method == 'BasicHD' or self.method == 'SemiHD':
            self.classify = nn.Linear(self.hd_dim, self.num_classes, bias=False)
            self.classify_sample_cnt = torch.zeros((self.num_classes, 1)).to(self.device)

        elif self.method == 'LifeHDsemi':
            # The first num_class in the classify is for labeled classes,
            # while the remaining are unlabeled prototypes
            self.classify = nn.Linear(self.hd_dim, self.max_classes, bias=False)
            self.classify_sample_cnt = torch.zeros(self.max_classes).to(self.device)
            self.dist_mean = torch.zeros(self.max_classes).to(self.device)
            self.dist_std = torch.zeros(self.max_classes).to(self.device)
            self.last_edit = - np.ones(self.max_classes)

        else:
            raise ValueError('method not supported: {}'.format(self.method))

        self.classify.weight.data.fill_(0.0)

        # self.classify_weights is the sum of all hypervectors, so its scale
        # accounts the number of samples in this class/cluster
        self.classify_weights = copy.deepcopy(self.classify.weight)
        # print(self.classify_weights.shape)  # size num_class x HD dim


    def encode(self, x, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.feature_extractor != 'none':
            x = self.net(x)

        x = self.flatten(x)
        sample_hv = torch.zeros((x.shape[0], self.hd_dim), device=self.device)

        if self.hd_encoder == 'rp':
            sample_hv[:, mask] = self.projection(x)[:, mask]

        elif self.hd_encoder == 'idlevel':
            # print(self.value(x)[:, :, mask].shape)  # (64, 561, 1000)
            # print(self.position.weight[:, mask].shape)  # (561, 1000)
            tmp_hv = functional.bind(self.position.weight[:, mask],
                                     self.value(x)[:, :, mask])  # bsz x num_features x hd_dim
            sample_hv[:, mask] = functional.multiset(tmp_hv)  # bsz x hd_dim

        elif self.hd_encoder == 'nonlinear':
            sample_hv[:, mask] = self.nonlinear_projection(x)[:, mask]

        elif self.hd_encoder == 'spatiotemporal':
            # First restore the time series order
            x = x.reshape((-1, self.win_size, self.input_dim))
            # tmp_hv = functional.bind(self.position.weight[:, mask],
            #                         self.value(x)[:, :, mask])  # bsz x num_features x hd_dim
            # Bundle
            # tmp_hv = functional.multiset(tmp_hv)  # bsz x T x dim
            # tmp_hv = functional.hard_quantize(tmp_hv)
            # Permutation and bind
            # sample_hv[:, mask] = functional.bind_sequence(tmp_hv)
            
            enc = []
            batch_size = x.shape[0]
            for i in range(batch_size):
                enc.append(
                    self.timeseries_Encoder.encode_one_time_series_sample(
                    x[i]).to(self.device))
            sample_hv[:, mask] = torch.stack(enc, dim=0)[:, mask]

        else:  # None encoder, just use the raw sample
            return x

        sample_hv[:, mask] = functional.hard_quantize(sample_hv[:, mask])
        return sample_hv

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        # Get logits output
        enc = self.encode(x, mask)
        # Compute the cosine distance between normalized hypervectors
        logits = self.classify(F.normalize(enc))

        #logits = torch.div(logits, self.temperature)
        #softmax_logits = F.log_softmax(logits, dim=1)

        return logits, enc # enc is still hd_dim, but some elements are 0

    def extract_class_hv(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        else:  # self.method == 'BasicHD'
            #class_hv = self.classify_weights / self.classify_sample_cnt
            class_hv = self.classify.weight[:, mask]
        return class_hv.detach().cpu().numpy()
    
    def extract_pair_simil(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD' or self.method == 'LifeHDsemi':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        elif self.method == 'BasicHD':
            class_hv = self.classify.weight[:, mask]
        else:
            raise ValueError('method not supported: {}'.format(self.method))
        pair_simil = class_hv @ class_hv.T

        if self.method == 'LifeHDsemi':
            pair_simil[:self.num_classes, :self.num_classes] = torch.eye(self.num_classes)
        return pair_simil.detach().cpu().numpy(), class_hv.detach().cpu().numpy()


def set_model(opt, num_classes, device):
    return Model(opt, num_classes, device)
