python pyscripts/mask2former_finetune.py \
    --config-file 'Mask2Former/configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml' \
    --num-gpus 1 --dist-url 'auto' \
    --num-machines 1 --data-mode placid \
    --cropped --wandb --wandb-name 'cityscapes-comic-50-swin_large-backbone' \
    --batch-size 8 --lr 0.0001 \
    --epochs 100 --eval-type 'comic_sem_seg' \
    --fn-mode 'decoder' \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    MODEL.WEIGHTS 'https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_large_IN21k_384_bs16_90k/model_final_17c1ee.pkl' \
    OUTPUT_DIR '/sinergia/shabanza/outputs/cityscapes'
