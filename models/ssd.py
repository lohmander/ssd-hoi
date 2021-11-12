import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from dataclasses import dataclass


@dataclass
class NPriorBoxes:
    fmap_1: int
    fmap_2: int
    fmap_3: int
    fmap_4: int
    fmap_5: int
    fmap_6: int


class BaseConv(nn.Module):
    def __init__(self):
        super().__init__()

        VGG16 = tv.models.vgg16(pretrained=True)

        # We extract two feature maps from the base module, the first
        # will be after 4 convs, the last will be after the last ones
        self.base_features = VGG16.features

        # Patch up the conv layer settings
        self.base_features[16].ceil_mode = True   # We ceil (insted of flooring) to keep the number of dims even
        self.base_features[30].kernel_size = 3    # We set the last layer kernel size to 3, to maintain the shape 
        self.base_features[30].padding = 1        # We set the last layer padding to 1, to maintain the shape 
        self.base_features[30].stride = 1         # We set the last layer stride to 1, to maintain the shape 

        # We also wanna include the classification layers of VGG16, however,
        # we want to convert the fully connected layers to also be cnns
        self.conv_features_fc1 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # atrous convolution
        self.conv_features_fc2 = nn.Conv2d(1024, 1024, kernel_size=1)

        # We'll need to tweak the state dicts for the linear layers to actually
        # work with our convolutional layers
        self.load_more_state()
    
    def load_more_state(self):
        state_dict = self.state_dict()
        classifier_state_dict = VGG16.classifier.state_dict()

        conv_fc6_weight = classifier_state_dict['0.weight'].view(4096, 512, 7, 7)
        conv_fc6_bias = classifier_state_dict['0.bias']
        state_dict['conv_features_fc1.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3]) # (1024, 512, 3, 3)
        state_dict['conv_features_fc1.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)

        conv_fc7_weight = classifier_state_dict['3.weight'].view(4096, 4096, 1, 1)
        conv_fc7_bias = classifier_state_dict['3.bias']
        state_dict['conv_features_fc2.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None]) # (1024, 1024, 1, 1)
        state_dict['conv_features_fc2.bias'] = decimate(conv_fc7_bias, m=[4]) # (1024

        self.load_state_dict(state_dict)
    
    def forward(self, x):
        # We extract two feature map representations from the base layer
        out = self.base_features[:23](x)
        fmap_1 = out

        out = self.base_features[23:](out)
        out = F.relu(self.conv_features_fc1(out))
        out = F.relu(self.conv_features_fc2(out))
        fmap_2 = out

        return fmap_1, fmap_2


class AuxConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        self.init_weights()
    
    def init_weights(self):
        for child in self.children():
            if not isinstance(child, nn.Conv2d):
                continue
            
            nn.init.xavier_uniform_(child.weight)
            nn.init.constant_(child.bias, 0.)

    def forward(self, fmap_x):
        out = F.relu(self.conv8_1(fmap_x))
        out = F.relu(self.conv8_2(out))
        fmap_3 = out
        
        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))

        fmap_4 = out

        out = F.relu(self.conv10_1(out))
        out = F.relu(self.conv10_2(out))
        fmap_5 = out


        out = F.relu(self.conv11_1(out))
        out = F.relu(self.conv11_2(out))
        fmap_6 = out

        return fmap_3, fmap_4, fmap_5, fmap_6


class PredConv(nn.Module):
    def __init__(self, n_cls):
        super().__init__()

        self.n_cls = n_cls

        bbox_points = 4
        n_boxes = NPriorBoxes(
            fmap_1=4,
            fmap_2=6,
            fmap_3=6,
            fmap_4=6,
            fmap_5=4,
            fmap_6=4,
        )

        # Localization convs
        self.loc_fmap_1 = nn.Conv2d(512, n_boxes.fmap_1 * bbox_points, kernel_size=3, padding=1)
        self.loc_fmap_2 = nn.Conv2d(1024, n_boxes.fmap_2 * bbox_points, kernel_size=3, padding=1)
        self.loc_fmap_3 = nn.Conv2d(512, n_boxes.fmap_3 * bbox_points, kernel_size=3, padding=1)
        self.loc_fmap_4 = nn.Conv2d(256, n_boxes.fmap_4 * bbox_points, kernel_size=3, padding=1)
        self.loc_fmap_5 = nn.Conv2d(256, n_boxes.fmap_5 * bbox_points, kernel_size=3, padding=1)
        self.loc_fmap_6 = nn.Conv2d(256, n_boxes.fmap_6 * bbox_points, kernel_size=3, padding=1)

        # Object class convs
        self.cls_fmap_1 = nn.Conv2d(512, n_boxes.fmap_1 * n_cls, kernel_size=3, padding=1)
        self.cls_fmap_2 = nn.Conv2d(1024, n_boxes.fmap_2 * n_cls, kernel_size=3, padding=1)
        self.cls_fmap_3 = nn.Conv2d(512, n_boxes.fmap_3 * n_cls, kernel_size=3, padding=1)
        self.cls_fmap_4 = nn.Conv2d(256, n_boxes.fmap_4 * n_cls, kernel_size=3, padding=1)
        self.cls_fmap_5 = nn.Conv2d(256, n_boxes.fmap_5 * n_cls, kernel_size=3, padding=1)
        self.cls_fmap_6 = nn.Conv2d(256, n_boxes.fmap_6 * n_cls, kernel_size=3, padding=1)

        self.init_weights()
    
    def init_weights(self):
        for child in self.children():
            # Use xavier init for all conv layers
            if isinstance(child, nn.Conv2d):
                nn.init.xavier_uniform_(child.weight)
                nn.init.constant_(child.bias, 0.)
    
    def forward(self, fmap_1, fmap_2, fmap_3, fmap_4, fmap_5, fmap_6):
        n = fmap_1.shape[0] # batch size

        # Localization regression
        loc_fmap_1 = self.loc_fmap_1(fmap_1).permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        loc_fmap_2 = self.loc_fmap_2(fmap_2).permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        loc_fmap_3 = self.loc_fmap_3(fmap_3).permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        loc_fmap_4 = self.loc_fmap_4(fmap_4).permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        loc_fmap_5 = self.loc_fmap_5(fmap_5).permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        loc_fmap_6 = self.loc_fmap_6(fmap_6).permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        # Object class prediciton
        cls_fmap_1 = self.cls_fmap_1(fmap_1).permute(0, 2, 3, 1).contiguous().view(n, -1, self.n_cls)
        cls_fmap_2 = self.cls_fmap_2(fmap_2).permute(0, 2, 3, 1).contiguous().view(n, -1, self.n_cls)
        cls_fmap_3 = self.cls_fmap_3(fmap_3).permute(0, 2, 3, 1).contiguous().view(n, -1, self.n_cls)
        cls_fmap_4 = self.cls_fmap_4(fmap_4).permute(0, 2, 3, 1).contiguous().view(n, -1, self.n_cls)
        cls_fmap_5 = self.cls_fmap_5(fmap_5).permute(0, 2, 3, 1).contiguous().view(n, -1, self.n_cls)
        cls_fmap_6 = self.cls_fmap_6(fmap_6).permute(0, 2, 3, 1).contiguous().view(n, -1, self.n_cls)

        return (
            # Concat all the localization bboxes and classifications in order
            torch.cat([loc_fmap_1, loc_fmap_2, loc_fmap_3, loc_fmap_4, loc_fmap_5, loc_fmap_6], dim=1),
            torch.cat([cls_fmap_1, cls_fmap_2, cls_fmap_3, cls_fmap_4, cls_fmap_5, cls_fmap_6], dim=1),
        )


class SSD(nn.Module):
    def __init__(self, n_cls):
        super().__init__()

        self.n_cls = n_cls

        self.base_conv = BaseConv()
        self.aux_conv = AuxConv()
        self.pred_conv = PredConv(n_cls)

        # Low level features like in fmap_1 have very large scales, and thus can be rescaled
        # so we'll create a learnable parameter for this
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1)) # 512 based on the number of params in fmap_1

        self.priors = self.create_prior_bboxes()


    def init_weights(self):
        # We initialize the rescale factor with 20, which is somewhat arbitrary, but 
        # shouldn't matter much since the value is learned to during training
        nn.init_constant_(self.rescale_factors, 20)
    
    def create_prior_bboxes(self):
        # Taken from the original SSD paper
        fmap_dims = {
            "fmap_1": 38,
            "fmap_2": 19,
            "fmap_3": 10,
            "fmap_4": 5,
            "fmap_5": 3,
            "fmap_6": 1,
        }
        obj_scales = {
            "fmap_1": .1,
            "fmap_2": .2,
            "fmap_3": .375,
            "fmap_4": .55,
            "fmap_5": .725,
            "fmap_6": .9,
        }
        ratios = {
            "fmap_1": [1., 2., 0.5],
            "fmap_2": [1., 2., 3., 0.5, 0.333],
            "fmap_3": [1., 2., 3., 0.5, 0.333],
            "fmap_4": [1., 2., 3., 0.5, 0.333],
            "fmap_5": [1., 2., 0.5],
            "fmap_6": [1., 2., 0.5],
        }

        fmaps = list(fmap_dims.keys())
        priors = []
        
        for k, (fmap, dims) in enumerate(fmap_dims.items()):
            for i in range(dims):
                for j in range(dims):
                    cx = (j + 0.5) / dims
                    cy = (i + 0.5) / dims

                    for ratio in ratios[fmap]:
                        scale = obj_scales[fmap]
                        priors.append(
                            [cx, cy, scale * sqrt(ratio), scale / sqrt(ratio)]
                        )

                        # For aspect ratios of 1, add an additional prior for the geometric mean
                        # of the scale of the current fmap and the next (if any)
                        if ratio == 1.:
                            try:
                                add_scale = sqrt(scale * obj_scales[fmaps[k + 1]])
                            except IndexError:
                                add_scale = 1.
                            
                            priors.append([cx, cy, add_scale, add_scale])
        
        return torch.FloatTensor(priors).to(device).clamp(0, 1)
    
    def forward(self, img):
        fmap_1, fmap_2 = self.base_conv(img)

        # Rescale the first fmap after l2 norm
        norm = fmap_1.pow(2).sum(dim=1, keepdim=True).sqrt()
        fmap_1 = fmap_1 / norm * self.rescale_factors

        fmap_3, fmap_4, fmap_5, fmap_6 = self.aux_conv(fmap_2)
        locs, clss = self.pred_conv(fmap_1, fmap_2, fmap_3, fmap_4, fmap_5, fmap_6)

        return locs, clss # , ints
    
    def detect_objects(self, locs, clss, min_score, max_overlap, top_k):
        batch_size = locs.size(0)
        n_priors = self.priors.size(0)
        cls_yhats = F.softmax(clss, dim=2)

        all_bboxes = []
        all_labels = []
        all_scores = []

        assert n_priors == locs.size(1)

        for i in range(batch_size):
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(locs[i], self.priors)
            )

            bboxes = []
            labels = []
            scores = []

            # Get the classification scores for this item in the batch
            cls_yhat = cls_yhats[i]

            # Get the highest class prediction score for each class and prior
            max_scores, best_score = cls_yhat.max(dim=1)
 
            # Class 0 is "background," hence we start at 1
            for cls in range(1, self.n_cls):
                cls_scores = cls_yhat[:, cls]
                above_min_scores = cls_scores > min_score
                n_above_min_scores = above_min_scores.sum().item()

                # Skip if none are greater than the threshold
                if n_above_min_scores == 0:
                    continue
                
                # Only keep the scores > min_score and their corresponding locs
                cls_scores = cls_scores[above_min_scores]
                cls_locs = decoded_locs[above_min_scores]

                # Sort the scores descendingly
                cls_scores, sort_idx = cls_scores.sort(dim=0, descending=True)
                cls_locs = cls_locs[sort_idx]

                # Find all the overlaps between bboxes
                overlap = find_iou(cls_locs, cls_locs)
                suppress = torch.zeros((n_above_min_scores,), dtype=torch.uint8).to(device)

                for box in range(cls_locs.size(0)):
                    # Skip if already marked for suppression?
                    if suppress[box] == 1:
                        continue
                    
                    suppress = torch.max(suppress, overlap[box] > max_overlap)

                    # Will have a 1.0 overlap with itself, but still we don't wanna suppress it
                    suppress[box] = 0 

                bboxes.append(cls_locs[1 - suppress])
                labels.append(torch.LongTensor((1 - suppress).sum().item() * [cls]).to(device))
                scores.append(cls_scores[1 - suppress])

            # If nothing is detected, add a background class
            if not bboxes:
                bboxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                labels.append(torch.LongTensor([0]).to(device))
                scores.append(torch.FloatTensor([0.]).to(device))
            
            # Concat into single tensors
            bboxes = torch.cat(bboxes, dim=0)
            labels = torch.cat(labels, dim=0)
            scores = torch.cat(scores, dim=0)

            if scores.size(0) > top_k:
                scores, sort_idx = scores.sort(dim=0, descending=True)
                scores = scores[:top_k]
                bboxes = bboxes[sort_idx][:top_k]
                labels = labels[sort_idx][:top_k]
            
            all_bboxes.append(bboxes)
            all_labels.append(labels)
            all_scores.append(scores)

        return all_bboxes, all_labels, all_scores
