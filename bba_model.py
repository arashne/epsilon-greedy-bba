from typing import Dict

import numpy as np


class BBA(object):
    def __init__(self, lower=3, upper=13.5):
        self.lower = lower
        self.upper = upper
        self.rng = np.random.default_rng()

    def select_video_format(self, obs: Dict[str, np.array]):
        sizes = obs["sizes"]
        ssims = obs["ssims"]
        buffer = obs["buffer"]
        rnd = obs["rnd"]
        assert sizes.shape == ssims.shape
        invalid_mask = np.logical_and(np.abs(sizes-3) < 1e-6, np.abs(ssims-0.85) < 1e-6)
        if rnd is True:
            valid_indices = [i for i in range(len(sizes)) if invalid_mask[i] == False]
            return int(self.rng.choice(valid_indices))
        size_arr_valid = np.ma.array(sizes, mask=invalid_mask)
        ssim_arr_valid = np.ma.array(ssims, mask=invalid_mask)
        min_choice = size_arr_valid.argmin()
        max_choice = size_arr_valid.argmax()
        if buffer < self.lower:
            act = min_choice
        elif buffer >= self.upper:
            act = max_choice
        else:
            ratio = (buffer - self.lower) / float(self.upper - self.lower)
            min_chunk = size_arr_valid[min_choice]
            max_chunk = size_arr_valid[max_choice]
            bitrate = ratio * (max_chunk - min_chunk) + min_chunk
            mask = np.logical_or(invalid_mask, size_arr_valid > bitrate)
            act = np.ma.array(ssim_arr_valid, mask=mask).argmax()
        return int(act)
