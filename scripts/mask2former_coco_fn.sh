DETECTRON2_DATASETS='/sinergia/shabanza/datasets' python Mask2Former/train_net.py \
    --num-gpus 1 --dist-url 'auto' \
    --num-machines 1 --config-file 'Mask2Former/configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml' \
    MODEL.WEIGHTS 'https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_large_IN21k_384_bs16_90k/model_final_17c1ee.pkl' \
    SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.00007 \
    INPUT.DATASET_MAPPER_NAME "mask_former_semantic" \
    TEST.EVAL_PERIOD 500
    # SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.00007 \
