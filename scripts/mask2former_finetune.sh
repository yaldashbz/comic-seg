python pyscripts/mask2former_finetune.py \
    --config-file 'Mask2Former/configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml' \
    --num-gpus 1 --dist-url 'auto' \
    --num-machines 1 --data-mode placid \
    --cropped --wandb --wandb-name 'cityscapes-coco-swin_large-backbone' \
    --batch-size 2 --lr 0.0001 \
    --epochs 30 --chkp-period 1000 \
    --eval-type 'comic_sem_seg' \
    TRAIN.FN_MODE 'backbone' \
    SOLVER.CHECKPOINT_PERIOD 200 \
    MODEL.WEIGHTS 'https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_large_IN21k_384_bs16_90k/model_final_17c1ee.pkl'
    # --max-iter 3000 --chkp-period 200 --steps 1500 2900 \
    # --wandb-name 'placid_fn_matching' \
