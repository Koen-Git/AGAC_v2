import os

from gym_minigrid.wrappers import ImgObsWrapper

from agac.agac import AGAC
from core.cmd_util import make_vec_env
from core.tf_util import linear_schedule

def run_minigrid(
    env_id: str,
    num_timesteps: int,
    seed: int,
    rnd_adv: bool,
    add_noise_old: bool,
    add_noise_normal: bool,
    n_frame_stack: int = 4,
    policy: str = "CnnPolicy",
    log_prefix: str = ""
):
    log_dir = "./logs/%s/AGAC_seed%s/%s_old%s_normal%s" % (env_id, seed, log_prefix, add_noise_old, add_noise_normal)
    os.makedirs(log_dir, exist_ok=True)
    model_dir = "./models/AGAG/%s_%s_%s" % (env_id, log_prefix, seed)
    os.makedirs(model_dir, exist_ok=True)
    n_steps = 2048
    if policy == "CnnLstmPolicy": n_steps = 256
    print(n_steps)
    env = make_vec_env(env_id, 1, seed, n_stack=n_frame_stack, monitor_dir=log_dir, wrapper_class=ImgObsWrapper)

    model = AGAC(policy, env, verbose=1, seed=seed, vf_coef=0.5, tensorboard_log=log_dir,
                     n_steps=n_steps, nminibatches=1, agac_c=linear_schedule(0.0004), beta_adv=0.00004,
                     learning_rate=0.0003, ent_coef=0.01, cliprange=0.2, rnd_adv=rnd_adv, add_noise_old=add_noise_old,
                     add_noise_normal=add_noise_normal)
    # model = model.load("./models/AGAG/MiniGrid-ObstructedMaze-Full-v0/ts_150001664.zip", env=env)
    model.learn(total_timesteps=num_timesteps, tb_log_name="tb/AGAC", save_interval=10000000, model_save_path=model_dir)
