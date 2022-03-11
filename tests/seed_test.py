import gordongames
import gym
from gordongames.envs.ggames.ai import even_line_match, cluster_match
from gordongames.envs.ggames.constants import DIRECTION2STR
from gordongames.oracles import GordonOracle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    render = True
    seed = 12345,
    kwargs = {
        "targ_range": (1,4),
        "grid_size": (13,9),
        "pixel_density": 3,
        "harsh": True,
    }
    np.random.seed(seed)
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
    ]
    for env_name in tqdm(env_names):
        print("Testing Env:", env_name)
        env = gym.make(env_name, **kwargs)
        env.seed(seed)
        prev_obs = env.reset()
        prev_obs, _, _, _ = env.step(0)
        for i in range(10):
            env.seed(seed)
            obs = env.reset()
            obs, _, _, _ = env.step(0)
            try:
                assert np.array_equal(prev_obs, obs)
            except:
                bar = -np.ones((len(obs), 1))
                cat = np.concatenate([prev_obs,bar,obs], axis=1)
                plt.imshow(cat)
                plt.show()
                print("error occurred")

