import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn as nn

import random

class VisionEncoder(nn.Module):
    def __init__(self, input_size = (512, 512)):
        super().__init__()

        self.input_size = input_size

        self.Faster_RCNN = fasterrcnn_resnet50_fpn(pretrained=True)
        self.Faster_RCNN.eval()
        for p in self.Faster_RCNN.parameters():
            p.requires_grad = False



        self.Faster_RCNN_Body = self.Faster_RCNN.backbone.body



        self.CacheLayers = {'layer2':'c3',
                       'layer3':'c4'}
        self.FeatureExtractor = IntermediateLayerGetter(self.Faster_RCNN_Body, return_layers=self.CacheLayers)
        self.ConvUpSample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.Conv1x1 = nn.Conv2d(in_channels=1536, out_channels=512, kernel_size=1)
        self.Conv1x1 = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.AdaptPool = nn.AdaptiveAvgPool2d((5,5))

    def train(self, mode: bool = True):
        super().train(mode)
        self.Faster_RCNN.eval()
        return self

    def forward(self, x):

        # self.Faster_RCNN = self.Faster_RCNN.to(x.device)
        # self.Conv1x1 = self.Conv1x1.to(x.device)
        # self.Faster_RCNN.eval()

        ##################################
        result = self.Faster_RCNN(x)
        images, _ = self.Faster_RCNN.transform(x)
        inter_result = self.FeatureExtractor(x)

        orig_h, orig_w = images.image_sizes[0]

        # images, _ = self.Faster_RCNN.transform(x)
        # features = self.Faster_RCNN.backbone(images.tensors)
        # inter_result = self.FeatureExtractor(features)

        # result = self.Faster_RCNN(images.tensors)
        ##################################



        inter_result['c4'] = self.ConvUpSample(inter_result['c4'])

        attributeBlob = torch.cat([inter_result['c3'], inter_result['c4']], dim=1)
        attributeBlob = self.Conv1x1(attributeBlob) # (batch_size, 512, 64, 64) -- > (batch_size, objects, 512, 5, 5)

        # print(attributeBlob.shape)

        batch_size = attributeBlob.shape[0]

        H, W = attributeBlob.shape[-2:]


        # print(result)

        objectsBlob = torch.zeros(size=(batch_size, 10, 512,5,5), device=x.device)
        new_boxes = []

        for sample in range(batch_size):
            num_objects = 10 if len(result[sample]['boxes']) > 10 else len(result[sample]['boxes'])
            boxes = result[sample]['boxes'][:num_objects]

            # print(boxes)
            # shuffled = None
            # if len(boxes) > 0:
            #     shuffled = random.sample(boxes, k=len(boxes))
            # else:
            #     shuffled = boxes
            perm = torch.randperm(boxes.size(0))
            shuffled = boxes[perm]

            new_boxes.append(shuffled)

            for obj in range(num_objects):
                bbox = shuffled[obj]

                scaled = bbox * (H/orig_h)

                x1, y1 = scaled[0].floor().clamp(0, W-1).type(torch.int64), scaled[1].floor().clamp(0, H-1).type(torch.int64)
                x2, y2 =  scaled[2].ceil().clamp(0, W-1).type(torch.int64), scaled[3].ceil().clamp(0, H-1).type(torch.int64)

                x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                if x2 <= x1:
                    x2 = min(x1 + 1, W)
                if y2 <= y1:
                    y2 = min(y1 + 1, H)

                object_features = attributeBlob[sample, :, y1:y2, x1:x2]


                objectsBlob[sample, obj, :, :, :] = self.AdaptPool(object_features)


        return new_boxes, objectsBlob
    
from copy import deepcopy
class LocationVisionEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        self.input_proj = nn.Linear(5, 512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=2,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu',
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.norm = nn.LayerNorm(512)

    def ProcessBoxes(self, boxes):
        boxes_copy = torch.zeros(size=(10, 5), device=self.device)

        for i, box in enumerate(boxes):
            w = abs(box[2] - box[0])
            h = abs(box[3] - box[1])
            boxes_copy[i,:] = torch.tensor(
                [box[0]/512, box[1]/512, box[2]/512, box[3]/512, (w*h) / (512*512)],
                device=self.device)

        return boxes_copy

    def forward(self, fastRCNN_Out):
        batch_size = len(fastRCNN_Out)
        loc_features = torch.zeros(size=(batch_size, 10,5), device=self.device)

        for i in range(batch_size):
            n = 10 if len(fastRCNN_Out[i]) > 10 else len(fastRCNN_Out[i])
            boxes = fastRCNN_Out[i][:n]
            # print(boxes.shape)
            # print(self.ProcessBoxes(boxes))
            loc_features[i,:,:] = self.ProcessBoxes(boxes)

        # print(loc_features)
        x = self.input_proj(loc_features)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)

        x = x.permute(1, 0, 2)

        return self.norm(x)