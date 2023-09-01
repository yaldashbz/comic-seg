python pyscripts/mask2former_finetune.py \
    --num-gpus 2 --dist-url 'auto' \
    --num-machines 1 --data-mode placid \
    --panel --wandb-name 'batch2_freeze_backbone' --batch-size 2
