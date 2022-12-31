import sys, os

base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(base_dir)

import yaml
from arguments import *
from utils.common import *

import algorithms.gd as gd

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    args_list = ['--cfg', './cfg/TactilePush/gd_tactile.yaml',
                 '--logdir', './trained_models/',
                 '--log-interval', '1',
                 '--save-interval', '50',
                 '--render-interval', '0',
                 '--seed', '0']
    
    solve_argv_conflict(args_list)

    parser = get_gd_parser()

    args = parser.parse_args(args_list + sys.argv[1:])

    # load config
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader = yaml.SafeLoader)
    
    if not args.no_time_stamp:
        args.logdir = os.path.join(args.logdir, get_time_stamp())

    args.train = not args.play

    vargs = vars(args)
    
    cfg["params"]["general"] = {}
    for key in vargs.keys():
        cfg["params"]["general"][key] = vargs[key]

    algo = gd.GD(cfg)

    if args.train:
        algo.train()
    else:
        algo.play(cfg)