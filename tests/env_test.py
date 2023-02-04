import gordongames
import gym
from gordongames.envs.ggames.ai import even_line_match, cluster_match
from gordongames.envs.ggames.constants import DIRECTION2STR
from gordongames.oracles import GordonOracle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

if __name__=="__main__":
    render = True
    save_figs = False
    n_episodes = 3
    delay = 2

    kwargs = {
        "targ_range": (2,5),
        "hold_outs": {},
        "grid_size": (17,31),
        "pixel_density": 1,
        "seed": int(time.time()),
        "harsh": True,
        "max_steps": None,
        "rand_pdb": False,
        "player_on_pile": True,
        "rand_timing": False,
        "timing_p": 0.8,
        "spacing_limit": 2,
        "sym_distr": True,
        "min_play_area": True,
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
        #"gordongames-v8",
        #"gordongames-v9",
        #"gordongames-v10",
        "gordongames-v11",
        "gordongames-v12",
    ]
    start_time = time.time()
    for env_name in env_names:
        print("Testing Env:", env_name)
        env = gym.make(env_name, **kwargs)
        env.seed(kwargs["seed"])
        oracle = GordonOracle(env_name)
        targ_distr = {
            i: 0 for i in range(
                0,kwargs["targ_range"][-1]+1
            )
        }
        rng = range(n_episodes)
        if not render: rng = tqdm(rng)
        for i in rng:
            obs = env.reset()
            n_steps = 1
            done = False
            targ_distr[env.controller.n_targs] += 1
            if render:
                print()
                print("starting new episode")
                print("step:", n_steps)
                print()
                #plt.imshow(prev_obs)
                #plt.show()
                env.render()
                time.sleep(delay)
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

                if save_figs:
                    img = obs
                    fig = plt.figure(figsize=(10,10))
                    plt.imshow(img)
                    plt.savefig("imgs/{}{}_img{}.png".format(env_name, i, n_steps))
                    plt.clf()

                n_steps += 1
                if render:
                    print("step:", n_steps)
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
                    time.sleep(delay)
                    env.render()
        print("Targ distr")
        print("n_targs, count")
        for k,v in targ_distr.items():
            print(k, v)
    print("Tot time:", time.time()-start_time)

