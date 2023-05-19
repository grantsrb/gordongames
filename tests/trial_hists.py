import gordongames
import gym
from gordongames.envs.ggames.ai import even_line_match, cluster_match
from gordongames.envs.ggames.constants import DIRECTION2STR
from gordongames.oracles import GordonOracle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import numpy as np

if __name__=="__main__":
    n_episodes = 10000

    kwargs = {
        "targ_range": (1,15),
        "hold_outs": {},
        "grid_size": (13,23),
        "pixel_density": 1,
        "seed": int(time.time()),
        "harsh": True,
        "max_steps": 96,
        "rand_pdb": True,
        "player_on_pile": True,
        "rand_timing": True,
        "timing_p": 0.9,
        "spacing_limit": 9,
        "zipf_exponent": 1,
        "min_play_area": True,
        "center_signal": True,
    }
    env_names = [
        #"gordongames-v0",
        #"gordongames-v1",
        #"gordongames-v2",
        #"gordongames-v3",
        "gordongames-v4",
        #"gordongames-v5",
        #"gordongames-v6",
        #"gordongames-v7",
        #"gordongames-v8",
        #"gordongames-v9",
        #"gordongames-v10",
    ]
    start_time = time.time()
    avg_steps = 0
    min_steps = np.inf
    max_steps = 0
    n_games = 0
    for env_name in env_names:
        print("Testing Env:", env_name)
        env = gym.make(env_name, **kwargs)
        env.seed(kwargs["seed"])
        oracle = GordonOracle(env_name)
        skip_distr = [0,0]
        contig_skips = [0,0]
        targ_distr = {
            i: 0 for i in range(
                kwargs["targ_range"][0],kwargs["targ_range"][-1]+1
            )
        }
        for i in tqdm(range(n_episodes)):
            obs = env.reset()
            n_steps = 1
            done = False
            targ_distr[env.controller.n_targs] += 1
            n_steps = 0
            while not done:
                n_steps += 1
                actn = oracle(env)
                prev_obs = obs
                obs, rew, done, info = env.step(actn)
                if info["is_animating"]:
                    skip_distr[info["skipped"]] += 1
                    if info["skipped"]:
                        if contig_skips[1]:
                            contig_skips[0] += 1
                        else:
                            contig_skips[1] = 1
                    elif contig_skips[1]:
                        contig_skips[1] = 0
                        contig_skips[0] += 1
            n_games += 1
            avg_steps += n_steps
            if n_steps > max_steps: max_steps = n_steps
            elif n_steps < min_steps: min_steps = n_steps
        print("Targ distr")
        print("n_targs, count")
        for k,v in targ_distr.items():
            print(k, v)
        print("\nAvg Step Count:", avg_steps/n_episodes)
        print("Min Step Count:", min_steps)
        print("Max Step Count:", max_steps)
        print("\nSkip Stats")
        print("0:", skip_distr[0])
        print("1:", skip_distr[1])
        s = np.sum(skip_distr) - n_episodes
        print("Skip p:", skip_distr[1]/s)
        print("Contig Skips:", contig_skips[0], "-- p:", contig_skips[0]/s)
        print()
        print()
    print("Tot time:", time.time()-start_time)

