# This RL-env simulator is adapted from:
# https://github.com/sagar-pa/abr_rl_test/blob/e03d209603cc241910e607015cac9e22684ffab5/tara_env.py

import os
from base_env import BaseEnv
from typing import Callable
import numpy as np
from bba_model import BBA

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

EPSILON = 0.01


class EpsilonGreedyBBA(BaseEnv):
    def setup_env(self, model_path: str) -> Callable:
        model = BBA()
        return model.select_video_format

    def process_env_info(self, env_info: dict) -> dict:
        # env_info["past_chunk"]["delay"] -> Seconds
        # env_info["past_chunk"]["size"] -> MegaBytes
        # env_info["past_chunk"]["ssim"] -> Unitless
        # env_info["sizes"] -> MegaBytes, first one is next chunk
        # env_info["ssims"] -> Unitless, first one is next chunk
        # env_info["buffer"] -> Seconds

        video_sizes = np.array(env_info["sizes"][0])
        ssim_indices = np.array(env_info["ssims"][0])
        buffer = env_info["buffer"]
        ts = env_info["ts"]

        assert video_sizes.ndim == 1
        assert video_sizes.shape == ssim_indices.shape

        rnd = False
        if np.random.default_rng(seed=int(ts)).random() < EPSILON:
            rnd = True

        obs = {"sizes": video_sizes, "ssims": ssim_indices, "buffer": buffer, "rnd": rnd}
        return obs
