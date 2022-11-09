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
    rnd_noise: bool,
    add_noise: bool,
    log_prefix: str = ""
):
    log_dir = "./logs/%s/AGAC_seed%s/%snoise%s_add%s" % (env_id, seed, log_prefix, rnd_noise, add_noise)
    os.makedirs(log_dir, exist_ok=True)
    model_dir = "./models/AGAG/%s" % (env_id)
    os.makedirs(model_dir, exist_ok=True)
    env = make_vec_env(env_id, 1, seed, monitor_dir=log_dir, wrapper_class=ImgObsWrapper)

    model = AGAC('CnnPolicy', env, verbose=1, seed=seed, vf_coef=0.5, tensorboard_log=log_dir,
                     n_steps=2048, nminibatches=8, agac_c=linear_schedule(0.0004), beta_adv=0.00004,
                     learning_rate=0.0003, ent_coef=0.01, cliprange=0.2, rnd_adv=rnd_adv, rnd_noise=rnd_noise, add_noise=add_noise)
    # model = model.load("./models/AGAG/MiniGrid-ObstructedMaze-Full-v0/ts_150001664.zip", env=env)
    model.learn(total_timesteps=num_timesteps, tb_log_name="tb/AGAC", save_interval=10000000, model_save_path=model_dir)
