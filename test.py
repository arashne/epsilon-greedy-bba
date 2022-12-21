#!/usr/bin/env python3

# This test script is adapted from:
# https://github.com/sagar-pa/abr_rl_test/blob/e03d209603cc241910e607015cac9e22684ffab5/test.py

# A default shebang is set, but it may not point to the expected python path
import sys
from epsilon_greedy_bba_env import EpsilonGreedyBBA


def main(args_env):
    env = EpsilonGreedyBBA(model_path="", server_address=args_env["server_address"])
    env.env_loop()


if __name__ == "__main__":
    args = {"name": sys.argv[1],
            "model_path": sys.argv[2],
            "server_address": sys.argv[3]}
    try:
        main(args)
    except:
        print('Something went wrong')
        sys.exit(0)
