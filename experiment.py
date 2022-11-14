import argparse 

from run_minigrid import run_minigrid

default_env = "MiniGrid-KeyCorridorS3R3-v0"
default_seed = 4949

parser = argparse.ArgumentParser(description='Experiment on AGAC')

parser.add_argument('--env', type=str, default=default_env, help='Gym environment.')
parser.add_argument('--timesteps', default=200000000, type=int, help='Number of timesteps to run.')
parser.add_argument('--seed', default=default_seed, type=int, help='Random seed (default: 4949).')
parser.add_argument('--policy', default="CnnPolicy", type=str, help='Use random adv.', choices=["CnnPolicy", "CnnLstmPolicy"])
parser.add_argument('--n_frame_stack', default=4, type=int, help='Amount of stacked frames.')
parser.add_argument('--rnd_adv', default=False, type=bool, help='Add Random adversary (default: False).')
parser.add_argument('--add_noise_old', default=False, type=bool, help='Add noise (default: False).')
parser.add_argument('--add_noise_normal', default=False, type=bool, help='Add noise with normal dist (default: False).')
parser.add_argument('--log_prefix', default="", type=str, help='Log prefix (default: "").')

if __name__ == '__main__':
    args = parser.parse_args()

    run_minigrid(
        env_id=args.env, 
        num_timesteps=args.timesteps, 
        seed=args.seed, 
        policy=args.policy,
        n_frame_stack=args.n_frame_stack,
        rnd_adv=args.rnd_adv,
        add_noise_old=args.add_noise_old,
        add_noise_normal=args.add_noise_normal,
        log_prefix=args.log_prefix
    )