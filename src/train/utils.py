def freeze_mask2former(model, distributed=False):
    if distributed:
        model = model.module
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.sem_seg_head.parameters():
        param.requires_grad = True
    
    # for param in model.sem_seg_head.predictor.mask_embed.parameters():
        # param.requires_grad = True
