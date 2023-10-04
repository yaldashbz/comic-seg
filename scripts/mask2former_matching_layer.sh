python pyscripts/mask2former_finetune.py \
    --config-file 'Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml' \
    --num-gpus 1 --dist-url 'auto' \
    --num-machines 1 --data-mode placid --keep-class-ids 24 25 26 27 \
    --cropped --wandb --wandb-name '4label-comic-matching_layer-placid' \
    --batch-size 32 --lr 0.0001 \
    --epochs 100 --eval-type 'comic_sem_seg' \
    --fn-mode 'matching_layer' \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    MODEL.WEIGHTS 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl' \
    OUTPUT_DIR '/sinergia/shabanza/outputs/coco_matching_layer'