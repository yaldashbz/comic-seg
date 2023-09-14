python pyscripts/mask2former_finetune.py \
    --config-file 'Mask2Former/configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml' \
    --num-gpus 1 --dist-url 'auto' \
    --num-machines 1 --data-mode placid \
    --cropped --wandb --wandb-name 'ade20k-comic-50-swin_large-backbone' \
    --batch-size 2 --lr 0.00008 \
    --epochs 50 --eval-type 'comic_sem_seg' \
    --fn-mode 'decoder' \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    MODEL.WEIGHTS 'https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_large_IN21k_384_bs16_160k_res640/model_final_6b4a3a.pkl' \
    OUTPUT_DIR '/sinergia/shabanza/outputs/ade20k'
    # --max-iter 3000 --chkp-period 200 --steps 1500 2900 \
    # --wandb-name 'placid_fn_matching' \
