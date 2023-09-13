import enum

class FNType(enum.Enum):
    MATCHING_LAYER = 'matching_layer'
    DECODER = 'decoder'


def _fn_decoder(model):
    # pixel decoder
    for param in model.sem_seg_head.predictor.parameters():
        param.requires_grad = True


def _fn_matching_layer(model):
    for param in model.sem_seg_head.predictor.class_embed.parameters():
        param.requires_grad = True


FN_MATCHER = {
    FNType.MATCHING_LAYER.value: _fn_matching_layer,
    FNType.DECODER.value: _fn_decoder
}


def freeze_mask2former(
    model, distributed: bool = False, 
    mode: FNType = FNType.MATCHING_LAYER
):
    if distributed:
        model = model.module
    for param in model.parameters():
        param.requires_grad = False
    
    print(f'Finetuning is in mode {mode}')
    FN_MATCHER[mode](model)
