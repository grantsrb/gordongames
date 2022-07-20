import gordongames
import gym
from gordongames.envs.ggames.ai import even_line_match, cluster_match
from gordongames.envs.ggames.constants import DIRECTION2STR
from gordongames.oracles import GordonOracle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

if __name__=="__main__":
    render = False
    kwargs = {
        "targ_range": (11,12),
        "hold_outs": {3},
        "grid_size": (27,15),
        "pixel_density": 3,
        "seed": 123456,
        "harsh": True,
        "max_steps": None,
        "rand_pdb": False,
    }
    env_names = [
        "gordongames-v0",
        "gordongames-v1",
        "gordongames-v2",
        "gordongames-v3",
        "gordongames-v4",
        "gordongames-v5",
        "gordongames-v6",
        "gordongames-v7",
        "gordongames-v8",
        "gordongames-v9",
    ]
    start_time = time.time()
    for env_name in env_names:
        print("Testing Env:", env_name)
        env = gym.make(env_name, **kwargs)
        env.seed(kwargs["seed"])
        oracle = GordonOracle(env_name)
        targ_distr = {
            i: 0 for i in range(
                kwargs["targ_range"][0],kwargs["targ_range"][-1]+1
            )
        }
        targ_distr[1] = 0
        targ_distr[2] = 0
        rng = range(5)
        if not render: rng = tqdm(rng)
        for i in rng:
            obs = env.reset()
            done = False
            targ_distr[env.controller.n_targs] += 1
            while not done:
                actn = oracle(env)
                if render:
                    print("Testing Env:", env_name)
                    print("Env is_grabbing:", env.is_grabbing)
                    if actn < 5:
                        print("actn:", DIRECTION2STR[actn])
                    else:
                        print("actn: GRAB")
                prev_obs = obs
                obs, rew, done, info = env.step(actn)
                if render:
                    print("done: ", done)
                    print("rew: ", rew)
                    for k in info.keys():
                        print(k, info[k])
                    print("mean luminance:", obs.mean())
                    print("max luminance:", obs.max())
                    print("min luminance:", obs.min())
                    print()
                    #plt.imshow(prev_obs)
                    #plt.show()
                    env.render()
        print("Targ distr")
        print("n_targs, count")
        for k,v in targ_distr.items():
            print(k, v)
    print("Tot time:", time.time()-start_time)

