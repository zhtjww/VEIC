# VLM-guided Explicit-Implicit Complementary Novel Class Semantic Learning for Few-Shot Object Detection

**Journal:** Expert Systems With Applications, 2024

## Requirements
* Python == 3.7.13
* PyTorch == 1.7.1
* Torchvision == 0.8.2
* Detectron2 == 0.3
* CUDA == 10.1

## Data Preparation
1. Please follow the instructions in [DeFRCN](https://github.com/er-muyue/DeFRCN) to prepare VOC2007, VOC2012, and vocsplit.
2. Copy all images from VOC2007 and VOC2012 into the `VEIC_data/VOCImages` directory:
    ```bash
    python -m datasets.VEIC_data.copy_imgs
    ```

## Weights Preparation
1. Please follow the instructions in DeFRCN to download imagenet pretrain weights.
2. We adopt the base-training weights provided by DeFRCN. Please download them and place them in the `voc_base_ckpt` folder. 
3. The directory structure should be organized as follows:
    ```
      ...
      ImageNetPretrained
        |--resnet101-5d3b4d8f.pth
      voc_base_ckpt
        |-- defrcn_det_r101_base1
        |----model_reset_surgery.pth
        |-- defrcn_det_r101_base2
        |----model_reset_surgery.pth
        |-- defrcn_det_r101_base3
        |----model_reset_surgery.pth
      ...
    ```

## Training and Evaluation

* To train and evaluate the model on PASCAL VOC, run the following command:
  ```
  bash run_voc.sh EXP_NAME SPLIT_ID 
  example: bash run_voc.sh VEIC 1
  ```
## Acknowledgement
This repo is developed based on [DeFRCN](https://github.com/er-muyue/DeFRCN), [Detectron2](https://github.com/facebookresearch/detectron2), [MFDC](https://github.com/shuangw98/MFDC) and [CLIP](https://github.com/openai/CLIP). We thank the authors for their valuable contributions.