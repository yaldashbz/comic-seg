python pyscripts/mask2former_finetune.py \
    --num-gpus 1 --dist-url 'auto' \
    --num-machines 1 --data-mode placid \
    --cropped --wandb --wandb-name 'test2' \
    --batch-size 6 --lr 0.0001 --fn-mode 0 \
    --epochs 20 --chkp-period 1000 \
    --eval-type 'comic_sem_seg'
    # --max-iter 3000 --chkp-period 200 --steps 1500 2900 \
    # --wandb-name 'placid_fn_matching' \
