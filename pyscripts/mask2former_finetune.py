import os
import sys

from torch.nn.parallel import DistributedDataParallel
from detectron2.engine import default_argument_parser, launch
import detectron2.utils.comm as comm


def cli():
    parser = default_argument_parser()
    parser.add_argument('--data-mode', default='placid')
    parser.add_argument('--wandb-name', default='mask2former_fn')
    parser.add_argument('--batch-size', default=4)
    parser.add_argument('--test-size', default=0.2)
    parser.add_argument('--random-state', default=42)
    parser.add_argument('--panel', action='store_true')
    args = parser.parse_args()
    return args


def main(args):
    import wandb
    from src.train import setup, ComicTrainer, do_train, do_test

    wandb.init(
        project="comic-seg",
        config={
            "panel_wise": args.panel,
            "dataset": args.data_mode,
        },
        name=args.wandb_name
    )
    cfg = setup(
        args.data_mode, args.panel, args.test_size, 
        args.random_state, args.batch_size
    )
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
    do_train(cfg, model, resume=args.resume, distributed=distributed)
    do_test(cfg, model)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'Mask2Former'))


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
