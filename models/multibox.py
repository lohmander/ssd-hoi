import torch
import torch.nn as nn
import sshoid.utils as utils


class MultiboxLoss(nn.Module):
    def __init__(self, priors, threshold=0., neg_pos_ratio=3, alpha=1.):
        super().__init__()

        self.priors_cxcy = priors
        self.priors_xy = utils.cxcy_to_xy(priors)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.smooth_l1 = nn.L1Loss()
        self.crossentropy = nn.CrossEntropyLoss(reduction='none')

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.priors_cxcy = self.priors_cxcy.to(*args, **kwargs) 
        self.priors_xy = self.priors_xy.to(*args, **kwargs) 
        self._device = args[0]
        print("set loss device", args[0])
        return self

    @property
    def device(self):
        return torch.device("cuda:0")
        # return self._device or next(self.parameters()).device

    def forward(self, yhat_locs, yhat_clss, y_locs, y_clss):
        batch_size = yhat_locs.size(0)
        n_priors = self.priors_xy.size(0)
        n_clss = yhat_clss.size(2)

        assert n_priors == yhat_locs.size(1) == yhat_clss.size(1), (n_priors, yhat_locs.shape, yhat_clss.shape)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)
        true_clss = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)

        for i in range(batch_size):
            n_objs = y_locs[i].size(0)
            overlap = utils.find_iou(y_locs[i], self.priors_xy)
            overlap_per_prior, object_per_prior = overlap.max(dim=0)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5)
            _, prior_per_object = overlap.max(dim=1)
            object_per_prior[prior_per_object] = torch.LongTensor(range(n_objs)).to(self.device)
            overlap_per_prior[prior_per_object] = 1.

            cls_per_prior = y_clss[i][object_per_prior] 
            cls_per_prior[overlap_per_prior < self.threshold] = 0

            true_locs[i] = utils.cxcy_to_gcxgcy(
                utils.xy_to_cxcy(y_locs[i][object_per_prior]),
                self.priors_cxcy,
            )
            true_clss[i] = cls_per_prior

            if torch.isnan(torch.sum(true_locs[i])):
                print(y_locs[i])
                print(xy_to_cxcy(y_locs[i]))
                raise Exception("torch is nan")

            if torch.isinf(torch.sum(true_locs[i])):
                print(y_locs[i])
                raise Exception("Err inf")
            
        pos_priors = true_clss != 0

        # Compute localization loss
        loc_loss = self.smooth_l1(yhat_locs[pos_priors], true_locs[pos_priors])

        if torch.isnan(loc_loss):
            print("pos priors", pos_priors)
            print("yhat", yhat_locs[pos_priors])
            print("y", true_locs[pos_priors])
            print("loc_loss", loc_loss)
            raise Exception("loc loss is nan")
        
        n_pos = pos_priors.sum(dim=1)
        n_hard_neg = self.neg_pos_ratio * n_pos


        # Compute the class confidence loss
        # Run loss for the pos priors and the hardest negative ones
        n_pos = pos_priors.sum(dim=1)
        n_hard_neg = self.neg_pos_ratio * n_pos

        conf_loss = self.crossentropy(
            yhat_clss.view(-1, n_clss),
            true_clss.view(-1),
        )

        conf_loss = conf_loss.view(batch_size, n_priors)
        conf_loss_pos = conf_loss[pos_priors]

        # Hard negatives
        conf_loss_neg = conf_loss.clone()
        conf_loss_neg[pos_priors] = 0.
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)

        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(self.device)
        hard_neg = hardness_ranks < n_hard_neg.unsqueeze(1)

        conf_loss_hard_neg = conf_loss_neg[hard_neg]
        conf_loss_total = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_pos.sum().float()
        
        return conf_loss_total + self.alpha * loc_loss
