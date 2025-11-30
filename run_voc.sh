#!/usr/bin/env bash

EXP_NAME=$1
SPLIT_ID=$2
SAVE_DIR=checkpoints/voc/${EXP_NAME}
IMAGENET_PRETRAIN_TORCH=ImageNetPretrained/resnet101-5d3b4d8f.pth
BASE_WEIGHT=voc_base_ckpt/defrcn_det_r101_base${SPLIT_ID}/model_reset_surgery.pth

for repeat in 0
do
    for shot in 1 2 3 5 10
    do
        for seed in 0
        do
            BASE_WEIGHT=voc_base_ckpt/defrcn_det_r101_base${SPLIT_ID}/model_reset_surgery.pth
            python3 tools/create_config.py --dataset voc --config_root configs/voc               \
                --shot ${shot} --seed ${seed} --setting 'gfsod' --split ${SPLIT_ID}
            CONFIG_PATH=configs/voc/defrcn_gfsod_r101_novel${SPLIT_ID}_${shot}shot_seed${seed}.yaml
            OUTPUT_DIR=${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}_repeat${repeat}
            OUTPUT_DIR_TEST=${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like/${shot}shot_seed${seed}_repeat${repeat}/inference
            if [ -d "$OUTPUT_DIR_TEST" ]; then
                echo "Results are available in ${OUTPUT_DIR_TEST}. Skip this."
            else
                python3 main.py --num-gpus 8 --config-file ${CONFIG_PATH}                            \
                    --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                     \
                           TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH} VOC_SPLIT ${SPLIT_ID}
#                rm ${CONFIG_PATH}
#                rm ${OUTPUT_DIR}/model_final.pth
            fi
        done
    done
done
python3 tools/extract_results.py --res-dir ${SAVE_DIR}/defrcn_gfsod_r101_novel${SPLIT_ID}/tfa-like --shot-list 1 2 3 5 10