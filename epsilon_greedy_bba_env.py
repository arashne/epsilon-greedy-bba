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

# OBS_HISTORY = 8
# NUM_CHOICES_SCALE = 12
# MAX_BUFFER_S = 15                   # Seconds
# REBUF_BASE_PEN = 60
# Q_BAR_BAR = 1 - np.power(10, -15.07/10)
# REBUF_DYN_PEN = REBUF_BASE_PEN / 10 * np.log(10) / 2.002 * (1 - Q_BAR_BAR)
# REBUF_DYN_PEN = 0.215
THR_SCALE = 1 / 8  # MB -> Mb
# SSIM_DB_SCALE = 60
# MAX_THR = 40                        # Mbps
# MAX_DTIME = 20                      # Seconds
MAX_CSIZE = 35  # Mb


# CXT_WIDTH = 3 * OBS_HISTORY + 3
# OBS_WIDTH = CXT_WIDTH + 2


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

        video_sizes = np.array(env_info["sizes"][0]) / THR_SCALE
        video_sizes = np.clip(video_sizes, a_max=MAX_CSIZE, a_min=0)
        ssim_indices = np.array(env_info["ssims"][0])
        buffer = env_info["buffer"]

        assert video_sizes.ndim == 1
        assert video_sizes.shape == ssim_indices.shape

        rnd = False
        if self.past_action is None:
            rnd = True
        else:
            small_ssim = ssim_indices[0]  # Is order preserved???
            if 0 < small_ssim < 1:
                seed = int(small_ssim * 2 ** 20)
                rng = np.random.default_rng(seed=seed)
                sample = rng.random()
                if sample < 0.01:
                    rnd = True

        obs = {"sizes": video_sizes, "ssims": ssim_indices, "buffer": buffer, "rnd": rnd}

        return obs
