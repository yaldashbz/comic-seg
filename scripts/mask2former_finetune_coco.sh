python pyscripts/mask2former_finetune.py \
    --config-file 'Mask2Former/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml' \
    --num-gpus 1 --dist-url 'auto' \
    --num-machines 1 --data-mode placid \
    --cropped --wandb --wandb-name 'coco-comic-50-swin_large-backbone' \
    --batch-size 2 --lr 0.00008 \
    --epochs 50 --eval-type 'comic_sem_seg' \
    --fn-mode 'decoder' \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    MODEL.WEIGHTS 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl' \
    OUTPUT_DIR '/sinergia/shabanza/outputs/coco'
    # --max-iter 3000 --chkp-period 200 --steps 1500 2900 \
    # --wandb-name 'placid_fn_matching' \
