import os
import sys

from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel
from detectron2.engine import default_argument_parser, launch
import detectron2.utils.comm as comm


def cli():
    parser = default_argument_parser()
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--data-mode', default='placid')
    parser.add_argument('--keep-class-ids', choices=list(range(28)), type=int, nargs='+')
    parser.add_argument('--wandb-name', default='mask2former_fn')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--cropped', action='store_true')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval-type', choices=[member.value for member in EvalType])
    parser.add_argument('--fn-mode', choices=[member.value for member in FNType])
    args = parser.parse_args()
    return args


def init_wandb(args):
    wandb.init(
        project="comic-seg",
        config={
            "panel_wise": args.cropped,
            "dataset": args.data_mode,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "model": "mask2former"
        },
        name=args.wandb_name
    )


def log_wandb(trainer, cfg):    
    log = defaultdict(list)
    window_size = cfg.ONE_EPOCH
    histories = trainer.storage.histories()
    for k, v in histories.items():
        val = v.values()
        if 'loss' in k:
            for i in range(0, len(val), window_size):
                window = val[i:i+window_size]
                window = [w[0] for w in window]
                avg = sum(window) / len(window)
                log[k].append(avg)
                wandb.log({k: avg})
    return log


def plain_main(args):
    from src.train import mask2former_setup, Mask2FormerComicTrainer, do_train, do_test

    cfg = mask2former_setup(args)
    print("Command Line Args:", args)
    model = Mask2FormerComicTrainer.build_model(cfg) 
    distributed = comm.get_world_size() > 1
    print('distributed: ', distributed)
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], 
            broadcast_buffers=False,
            find_unused_parameters=True
        ) 
    do_train(cfg, model, mode=args.TRAIN.FN_MODE, resume=args.resume, distributed=distributed, wandb_en=args.wandb)
    do_test(cfg, model)


def main(args):
    from src.train import mask2former_setup, Mask2FormerComicTrainer
    
    cfg = mask2former_setup(args)
    print("Command Line Args:", args)    
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Mask2FormerComicTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    if args.wandb:
        log_wandb(trainer, cfg)
        wandb.finish()


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'Mask2Former'))

    import wandb
    from src.dataset.helpers import EvalType
    from src.train.utils import FNType

    args = cli()
    print("Command Line Args:", args)
    if args.wandb:
        init_wandb(args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
