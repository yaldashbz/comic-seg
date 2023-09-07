python pyscripts/mask2former_finetune.py \
    --num-gpus 1 --dist-url 'auto' \
    --num-machines 1 --data-mode placid \
    --cropped --wandb-name 'placid_fn_matching' \
    --batch-size 1 --lr 0.0005 --fn-mode 0 \
    --eval-type 'comic_sem_seg'
