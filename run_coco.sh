#!/usr/bin/env bash

EXPNAME=$1
SAVEDIR=checkpoints/coco/${EXPNAME}
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/resnet101-5d3b4d8f.pth
BASE_WEIGHT=coco_base_ckpt/defrcn_det_r101_base/model_reset_surgery.pth

for repeat in 0
do
    for shot in 1 2 3 5 10 30
    do
        CONFIG_PATH=configs/coco/defrcn_gfsod_r101_novel_${shot}shot_seed0.yaml
        OUTPUT_DIR=${SAVEDIR}/defrcn_gfsod_r101_novel/tfa-like/${shot}shot_seed0
        OUTPUT_DIR_TEST=${SAVEDIR}/defrcn_gfsod_r101_novel/tfa-like/${shot}shot_seed0/inference
        if [ -d "$OUTPUT_DIR_TEST" ]; then
            echo "Results are available in ${OUTPUT_DIR_TEST}. Skip this."
        else
            python3 main.py --num-gpus 8 --config-file ${CONFIG_PATH}                      \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT}   OUTPUT_DIR ${OUTPUT_DIR}               \
                       TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}
            rm ${OUTPUT_DIR}/model_final.pth
        fi
    done
done
python3 tools/extract_results.py --res-dir ${SAVEDIR}/defrcn_gfsod_r101_novel/tfa-like --shot-list 1 2 3 5 10 30