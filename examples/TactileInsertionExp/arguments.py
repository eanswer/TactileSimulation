import argparse

def get_rl_parser():
    parser = argparse.ArgumentParser(description='RL')
    
    parser.add_argument(
        '--cfg', type=str, default='./cfg/tactile_insertion_trans_and_rot.yaml',
        help='specify the config file for the run'
    )

    parser.add_argument(
        '--play', action='store_true',
        help='play a stored policy'
    )

    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help="restore the policy from a checkpoint"
    )

    parser.add_argument(
        '--record', action="store_true",
        help="whether to export video for render during play mode"
    )

    parser.add_argument(
        '--logdir', type=str, default="./trained_models/ppo/",
        help="directory of logging"
    )

    parser.add_argument(
        '--no-time-stamp', action='store_true',
        help='whether not to add the timestamp in log folder'
    )

    parser.add_argument(
        '--save-interval', type=int, default=50,
        help="interval between two saved checkpoints"
    )

    parser.add_argument(
        '--seed', type=int, default=0,
        help="random seed"
    )

    parser.add_argument(
        '--render-interval', type=int, default=20,
        help="interval between two policy rendering"
    )

    parser.add_argument(
        '--log-interval', type=int, default=1,
        help="interval between two logging item"
    )

    parser.add_argument(
        '--device', type=str, default='cpu',
        help="which device to use"
    )

    parser.add_argument(
        '--stochastic', action='store_true',
        help="whether to use stochastic policy during play"
    )
    
    parser.add_argument(
        '--num-games', type=int, default=-1,
        help="number of games for evaluation"
    )

    parser.add_argument(
        '--render', action='store_true',
        default=False,
        help="whether to render during evaluation"
    )

    return parser