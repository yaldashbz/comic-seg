python pyscripts/deeplab_finetune.py \
    --config-file 'detectron2/projects/DeepLab/configs/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16.yaml' \
    --num-gpus 2 --dist-url 'auto' \
    --num-machines 1 --data-mode placid --keep-class-ids 24 25 26 27 \
    --cropped  \
    --batch-size 32 --lr 0.0001 \
    --epochs 100 --eval-type 'comic_sem_seg' \
    --fn-mode 'predictor' \
    SOLVER.CHECKPOINT_PERIOD 1000 \
    MODEL.WEIGHTS 'https://dl.fbaipublicfiles.com/detectron2/DeepLab/Cityscapes-SemanticSegmentation/deeplab_v3_plus_R_103_os16_mg124_poly_90k_bs16/28054032/model_final_a8a355.pkl' \
    OUTPUT_DIR '/sinergia/shabanza/outputs/deeplab'
    # --wandb --wandb-name 'deeplab-comic-predictor-200'
