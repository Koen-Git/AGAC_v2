import os

from gym_minigrid.wrappers import ImgObsWrapper

from agac.agac import AGAC
from core.cmd_util import make_vec_env
from core.tf_util import linear_schedule

num_envs = 1
num_minibatches = 8 * num_envs
rnd_noise = True
add_noise = True

for env_id in ["MiniGrid-ObstructedMaze-1Q-v0"]:
    for seed in [4949]:
        log_dir = "/data/s1861581/agac_results/logs/%s/with_noise%s_add%s/AGAC_%s_seed%s" % (env_id, rnd_noise, add_noise, num_envs, seed)
        os.makedirs(log_dir, exist_ok=True)
        model_dir = "./models/AGAG/%s" % (env_id)
        os.makedirs(model_dir, exist_ok=True)
        env = make_vec_env(env_id, num_envs, seed, monitor_dir=log_dir, wrapper_class=ImgObsWrapper)
        model = AGAC('CnnPolicy', env, verbose=1, seed=seed, vf_coef=0.5, tensorboard_log=log_dir,
                     n_steps=2048, nminibatches=num_minibatches, agac_c=linear_schedule(0.0004), beta_adv=0.00004,
                     learning_rate=0.0003, ent_coef=0.01, cliprange=0.2, rnd_noise=rnd_noise, add_noise=add_noise)
        model.learn(total_timesteps=200000000, tb_log_name="tb/AGAC", save_interval=10000000, model_save_path=model_dir)
