python pyscripts/mask2former_finetune.py \
    --num-gpus 1 --dist-url 'auto' \
    --num-machines 1 --data-mode placid \
    --cropped --wandb-name 'placid_fn_matching' \
    --batch-size 32 --lr 0.00005 --fn-mode 0 \
    --max-iter 1000 --chkp-period 200 \
    --eval-type 'comic_sem_seg'
