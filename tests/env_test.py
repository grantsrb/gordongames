import gordongames
import gym
from gordongames.envs.ggames.ai import even_line_match, cluster_match
from gordongames.envs.ggames.constants import DIRECTION2STR
from gordongames.oracles import GordonOracle

if __name__=="__main__":
    kwargs = {
        "targ_range": (1,6),
        "grid_size": (13,9),
        "pixel_density": 4,
        "harsh": True,
    }
    env_names = [
        "gordongames-v1",
        "gordongames-v2",
        "gordongames-v3",
        "gordongames-v5",
        "gordongames-v6",
    ]
    for env_name in env_names:
        print("Testing Env:", env_name)
        env = gym.make(env_name, **kwargs)
        oracle = GordonOracle(env_name)
        for i in range(7):
            obs = env.reset()
            done = False
            while not done:
                print("Testing Env:", env_name)
                actn = oracle(env)
                if actn < 5:
                    print("actn:", DIRECTION2STR[actn])
                else:
                    print("actn: GRAB")
                obs, rew, done, info = env.step(actn)
                print("done: ", done)
                print("rew: ", rew)
                print("grab", info["grab"])
                env.render()

