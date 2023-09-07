import os
import sys

from torch.nn.parallel import DistributedDataParallel
from detectron2.engine import default_argument_parser, launch
import detectron2.utils.comm as comm


def cli():
    parser = default_argument_parser()
    parser.add_argument('--data-mode', default='placid')
    parser.add_argument('--wandb-name', default='mask2former_fn')
    parser.add_argument(
        '--fn-mode', 
        default=0,
        type=int,
        choices=[0, 1, 2, 3]
    )
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--cropped', action='store_true')
    parser.add_argument('--eval-type', choices=[member.value for member in EvalType], nargs='+')
    args = parser.parse_args()
    return args


def main(args):
    import wandb
    from src.train import setup, ComicTrainer, do_train, do_test

    wandb.init(
        project="comic-seg",
        config={
            "panel_wise": args.cropped,
            "dataset": args.data_mode,
            "lr": args.lr,
            "batch_size": args.batch_size
        },
        name=args.wandb_name
    )
    cfg = setup(args)
    print("Command Line Args:", args)
    model = ComicTrainer.build_model(cfg) 
    distributed = comm.get_world_size() > 1
    print('distributed: ', distributed)
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], 
            broadcast_buffers=False,
            find_unused_parameters=True
        ) 
    do_train(cfg, model, mode=args.fn_mode, resume=args.resume, distributed=distributed)
    do_test(cfg, model)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'Mask2Former'))

    from src.dataset.helpers import EvalType

    args = cli()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
