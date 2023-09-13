python pyscripts/mask2former_finetune.py \
    --num-gpus 1 --dist-url 'auto' \
    --num-machines 1 --data-mode placid \
    --cropped --wandb --wandb-name 'test_semseg_predictor' \
    --batch-size 2 --lr 0.0001 --fn-mode 3 \
    --epochs 30 --chkp-period 1000 \
    --eval-type 'comic_sem_seg'
    # --max-iter 3000 --chkp-period 200 --steps 1500 2900 \
    # --wandb-name 'placid_fn_matching' \
