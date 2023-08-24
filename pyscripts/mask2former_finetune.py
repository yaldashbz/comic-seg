import os
import sys

from detectron2.engine import default_argument_parser, launch


def cli():
    import argparse

    parser = default_argument_parser()
    parser.add_argument('--dataset-name', default='placid')
    parser.add_argument('--test-size', default=0.2)
    parser.add_argument('--random-state', default=42)
    args = parser.parse_args()
    return args


def main(args):
    args.dataset_name = NAME_MAPPER[args.dataset_name]
    cfg = setup(args)
    print("Command Line Args:", args)    
    model = ComicTrainer.build_model(cfg)
    do_train(cfg, model, resume=args.resume)
    do_test(cfg, model)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'Mask2Former'))


    from src.train import *
    from src.dataset import NAME_MAPPER

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
