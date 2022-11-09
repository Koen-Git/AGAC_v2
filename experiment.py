import argparse 

from run_minigrid import run_minigrid


default_env = "MiniGrid-ObstructedMaze-1Q-v0"
default_seed = 4949

parser = argparse.ArgumentParser(description='Experiment on AGAC')

parser.add_argument('--env', type=str, default=default_env, help='Gym environment.')
parser.add_argument('--timesteps', default=200000000, type=int, help='Number of timesteps to run.')
parser.add_argument('--seed', default=default_seed, type=int, help='Random seed (default: 4949).')
parser.add_argument('--rnd_noise', default=False, type=bool, help='Random noise (default: False).')
parser.add_argument('--add_noise', default=False, type=bool, help='Add noise (default: False).')
parser.add_argument('--log_prefix', default="", type=str, help='Log prefix (default: "").')

if __name__ == '__main__':
    args = parser.parse_args()

    run_minigrid(
        env_id=args.env, 
        num_timesteps=args.timesteps, 
        seed=args.seed, 
        rnd_noise=args.rnd_noise,
        add_noise=args.add_noise,
        log_prefix=args.log_prefix
    )