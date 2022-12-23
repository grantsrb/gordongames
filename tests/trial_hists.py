import gordongames
import gym
from gordongames.envs.ggames.ai import even_line_match, cluster_match
from gordongames.envs.ggames.constants import DIRECTION2STR
from gordongames.oracles import GordonOracle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

if __name__=="__main__":
    n_episodes = 10000

    kwargs = {
        "targ_range": (1,17),
        "hold_outs": {},
        "grid_size": (13,29),
        "pixel_density": 1,
        "seed": int(time.time()),
        "harsh": True,
        "max_steps": 96,
        "rand_pdb": True,
        "player_on_pile": True,
        "rand_timing": False,
        "timing_p": 0.8,
        "spacing_limit": None,
        "zipf_exponent": 1,
        "min_play_area": True,
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
    n_games = 0
    for env_name in env_names:
        print("Testing Env:", env_name)
        env = gym.make(env_name, **kwargs)
        env.seed(kwargs["seed"])
        oracle = GordonOracle(env_name)
        targ_distr = {
            i: 0 for i in range(
                1,kwargs["targ_range"][-1]+1
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
            n_games += 1
            avg_steps += n_steps
        print("Targ distr")
        print("n_targs, count")
        for k,v in targ_distr.items():
            print(k, v)
        print("\nAvg Step Count:", avg_steps/n_episodes)
        print()
        print()
    print("Tot time:", time.time()-start_time)

