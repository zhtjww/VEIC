import torch
import logging
import random
import torch.nn.functional as F
from torch import nn
from detectron2.structures import ImageList
from detectron2.modeling.poolers import ROIPooler
from defrcn.data.class_names import PASCAL_VOC_ALL_CATEGORIES, COCO_ALL_CATEGORIES
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from defrcn.modeling.roi_heads import build_roi_heads
from defrcn.modeling.clip import clip

__all__ = ["GeneralizedRCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)

        self.clip, self.preprocess = clip.load('RN101', device="cpu")
        self.roi_pooler_clip = ROIPooler(output_size=(14, 14), scales=(1 / 16,), sampling_ratio=(0),
                                         pooler_type="ROIAlignV2")

        self.to(self.device)

        self.clip_text_fea = self.generate_clip_text()
        self.clip.half()

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        _, _, results, image_sizes = self._forward_once_(batched_inputs, None)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None):

        box_self = [i['instances'].gt_boxes.to(self.device) for i in batched_inputs]
        cls_self = [i['instances'].gt_classes.to(self.device) for i in batched_inputs]

        images = self.preprocess_image(batched_inputs)

        # for VEIC
        clip_mean = torch.tensor([0.406, 0.456, 0.485]).reshape((3, 1, 1)).to(self.device)
        clip_std = torch.tensor([[0.225, 0.224, 0.229]]).reshape((3, 1, 1)).to(self.device)

        fewshot_len = int(len(cls_self) * 0.75)
        images_e_clip = torch.zeros_like(images.tensor).float()
        features_det_extra = None

        if self.training:

            # re-normalize image for clip
            for i in range(images.tensor.size(0)):
                images_e_clip[i, :, :, :] = ((images.tensor[i, :, :, :] + torch.tensor(
                    [103.530, 116.280, 123.675]).reshape(3, 1, 1).to(images.tensor.device)) / 255.0 - clip_mean) / clip_std

            # extract res4 fea of detector for implicit pseudo set
            features_det_extra = self.backbone(images.tensor[fewshot_len:])
            features_det_extra = {
                k: self.affine_rcnn(decouple_layer(features_det_extra[k], self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE)) for
                k in features_det_extra}

            # keep original few-shot set and explict pseudo set
            images.image_sizes = images.image_sizes[:fewshot_len]
            images.tensor = images.tensor[:fewshot_len]
            gt_instances = gt_instances[:fewshot_len]

        # background exchange augmentation
        if self.training and random.random() > 0.5:
            images.tensor, gt_instances = self.shuffle_instance(images.tensor, box_self[:fewshot_len], gt_instances)

        features = self.backbone(images.tensor)
        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances,
                                                  images_e_clip=images_e_clip, features_det_extra=features_det_extra,
                                                  box_self=box_self, cls_self=cls_self,
                                                  clip_model=self.clip, clip_text_fea=self.clip_text_fea
                                                  )

        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std

    @staticmethod
    def shuffle_instance(x, box, gt_instance):
        device = x.device
        region_crop = []

        # cut instances
        for i in range(len(box)):
            for j in range(box[i].tensor.size(0)):
                x1, y1, x2, y2 = int(box[i].tensor[j][0]), int(box[i].tensor[j][1]), int(box[i].tensor[j][2]), int(
                    box[i].tensor[j][3])

                if y2 <= y1 or x2 <= x1 or y1 < 0 or x1 < 0 or y2 > x.size(2) or x2 > x.size(3):
                    continue

                region_crop_per = F.adaptive_avg_pool2d(x[i, :, y1:y2, x1:x2], (512, 512))
                region_crop.append(region_crop_per.unsqueeze(0))

        if not region_crop:
            return x, gt_instance

        # shuffle instance
        region_crop = torch.cat(region_crop, dim=0)
        index = torch.randperm(region_crop.size(0), device=device)
        region_crop = region_crop[index, :, :, :]
        cls_shuffle = torch.cat([i.gt_classes for i in gt_instance])[index]

        count = 0
        for i in range(x.size(0)):
            for j in range(box[i].tensor.size(0)):
                if count >= region_crop.size(0):
                    break

                x1, y1, x2, y2 = int(box[i].tensor[j][0]), int(box[i].tensor[j][1]), int(box[i].tensor[j][2]), int(
                    box[i].tensor[j][3])

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(x2, x.size(-1))
                y2 = min(y2, x.size(-2))

                if y2 <= y1 or x2 <= x1:
                    count += 1
                    continue

                x[i, :, y1:y2, x1:x2] = F.adaptive_avg_pool2d(
                    region_crop[count], (y2 - y1, x2 - x1)
                )
                gt_instance[i].gt_classes[j] = cls_shuffle[count]
                count += 1

        return x, gt_instance

    def generate_clip_text(self):
        templates = "this is a photo of a {}"
        if self.cfg.VOC_SPLIT != 0:
            cls_names = PASCAL_VOC_ALL_CATEGORIES[self.cfg.VOC_SPLIT]
        else:
            cls_names = COCO_ALL_CATEGORIES

        prompts = clip.tokenize([
            templates.format(cls_name)
            for cls_name in cls_names
        ]).to(self.device)
        with torch.no_grad():
            text_features = self.clip.encode_text(prompts)
            text_features.float()
        del prompts

        return text_features
