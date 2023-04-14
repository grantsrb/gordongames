import gordongames
import gym
from gordongames.envs.ggames.ai import even_line_match, cluster_match
from gordongames.envs.ggames.constants import DIRECTION2STR
from gordongames.oracles import GordonOracle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from collections import defaultdict

if __name__=="__main__":
    n_episodes = 1000
    held_out = True

    kwargs = {
        "targ_range": (1,5),
        "hold_outs": {},
        "grid_size": (10,10),
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
        "n_held_outs": 15,
    }
    env_names = [
        #"gordongames-v0",
        #"gordongames-v1",
        #"gordongames-v2",
        #"gordongames-v3",
        #"gordongames-v4",
        #"gordongames-v5",
        #"gordongames-v6",
        #"gordongames-v7",
        "gordongames-v8",
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
        rng = range(1,kwargs["targ_range"][-1]+1)
        targ_distr = {
          i: defaultdict(lambda: 0) for i in rng
        }
        for i in tqdm(range(n_episodes)):
            obs = env.reset(held_out=held_out)
            for t,targ in enumerate(env.controller.register.targs):
                targ_distr[t+1][targ.coord] += 1
            n_steps = 1
            done = False
            n_steps = 0
            while not done:
                n_steps += 1
                actn = oracle(env)
                prev_obs = obs
                obs, rew, done, info = env.step(actn)
            n_games += 1
            avg_steps += n_steps
        for k,v in targ_distr.items():
            print("Targ Quantity", k)
            print("Held Outs:", env.controller.register.held_outs[k])
            held_outs = env.controller.register.held_outs[k]
            for coord,count in v.items():
                print(coord, count)
                if held_out:
                    assert coord in held_outs
                else:
                    assert coord not in held_outs
            print()
        print()
        print()
    print("Tot time:", time.time()-start_time)

