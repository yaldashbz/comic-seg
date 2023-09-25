import enum

class FNType(enum.Enum):
    MATCHING_LAYER = 'matching_layer'
    DECODER = 'decoder'
    QUERY_FEAT = 'queryfeat'
    PREDICTOR = 'predictor'


def _fn_decoder(model):
    # pixel decoder
    for param in model.sem_seg_head.predictor.parameters():
        param.requires_grad = True


def _fn_matching_layer(model):
    for param in model.sem_seg_head.predictor.class_embed.parameters():
        param.requires_grad = True


def _fn_query_feat(model):
    _fn_matching_layer(model)
    for param in model.sem_seg_head.predictor.query_feat.parameters():
        param.requires_grad = True


def _fn_predictor(model):
    for param in model.sem_seg_head.predictor.parameters():
        param.requires_grad = True


FN_MATCHER_MASK2FORMER = {
    FNType.MATCHING_LAYER.value: _fn_matching_layer,
    FNType.DECODER.value: _fn_decoder,
    FNType.QUERY_FEAT.value: _fn_query_feat
}


FN_MATCHER_DEEPLAB = {
    FNType.PREDICTOR.value: _fn_predictor
}


def freeze_mask2former(
    model, mode: FNType = FNType.MATCHING_LAYER.value
):
    for param in model.parameters():
        param.requires_grad = False
    
    print(f'Finetuning is in mode {mode}')
    FN_MATCHER_MASK2FORMER[mode](model)


def freeze_deeplab(
    model, mode: FNType = FNType.MATCHING_LAYER.value
):
    for param in model.parameters():
        param.requires_grad = False
    
    print(f'Finetuning is in mode {mode}')
    FN_MATCHER_DEEPLAB[mode](model)
