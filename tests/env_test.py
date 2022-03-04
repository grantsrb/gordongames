import gordongames
import gym
from gordongames.envs.ggames.ai import even_line_match, cluster_match
from gordongames.envs.ggames.constants import DIRECTION2STR
from gordongames.oracles import GordonOracle
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__=="__main__":
    render = False
    kwargs = {
        "targ_range": (1,9),
        "grid_size": (11,11),
        "pixel_density": 3,
        "harsh": True,
    }
    env_names = [
        #"gordongames-v0",
        "gordongames-v1",
        #"gordongames-v2",
        #"gordongames-v3",
        #"gordongames-v4",
        #"gordongames-v5",
        #"gordongames-v6",
        #"gordongames-v7",
        #"gordongames-v8",
    ]
    for env_name in env_names:
        print("Testing Env:", env_name)
        env = gym.make(env_name, **kwargs)
        env.seed(1234)
        oracle = GordonOracle(env_name)
        targ_distr = {i: 0 for i in range(1,10)}
        rng = range(10000)
        if not render: rng = tqdm(rng)
        for i in rng:
            obs = env.reset()
            done = False
            targ_distr[env.controller.n_targs] += 1
            while not done:
                actn = oracle(env)
                if render:
                    print("Testing Env:", env_name)
                    if actn < 5:
                        print("actn:", DIRECTION2STR[actn])
                    else:
                        print("actn: GRAB")
                obs, rew, done, info = env.step(actn)
                if render:
                    print("done: ", done)
                    print("rew: ", rew)
                    print("grab", info["grab"])
                    print("n_targs:", info["n_targs"])
                    #plt.imshow(obs)
                    #plt.show()
                    env.render()
        print("Targ distr")
        print("n_targs, count")
        for k,v in targ_distr.items():
            print(k, v)

