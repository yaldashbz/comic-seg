import enum

class FNType(enum.Enum):
    MATHING_LAYER = 0
    DECODER = 3


def _fn_decoder(model):
    for param in model.sem_seg_head.parameters():
        param.requires_grad = True


def _fn_matching_layer(model):
    for param in model.sem_seg_head.predictor.class_embed.parameters():
        param.requires_grad = True


FN_MATCHER = {
    FNType.MATHING_LAYER.value: _fn_matching_layer,
    FNType.DECODER.value: _fn_decoder
}


def freeze_mask2former(
    model, distributed: bool = False, 
    mode: FNType = FNType.MATHING_LAYER
):
    if distributed:
        model = model.module
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    print(f'Finetuning is in mode {mode}')
    FN_MATCHER[mode](model)
