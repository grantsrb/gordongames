import gordongames
from gordongames.envs.ggames.constants import *
import gym
import time

if __name__=="__main__":
    args = {
        "targ_range": (5,6),
        "hold_outs": {},
        "grid_size": (10,10),
        "pixel_density": 1,
        "seed": int(time.time()),
        "harsh": True,
        "max_steps": None,
        "rand_pdb": False,
        "player_on_pile": True,
        "rand_timing": True,
        "timing_p": 0.5,
        "spacing_limit": 2,
        "sym_distr": True,
        "min_play_area": True,
        "n_held_outs": 15,
        "center_signal": False,
    }
    print("Seed:", args["seed"])
    env_types = [
        #"gordongames-v4",
        "gordongames-v8",
        #"gordongames-v10",
        #"gordongames-v11",
        #"gordongames-v12"
    ]

    print("PRESS q TO MOVE ON TO NEXT GAME TYPE")
    print("wasd to move, f to press")
    for env_type in env_types:
        print("making new env", env_type)
        env = gym.make(env_type, **args)

        done = False
        rew = 0
        obs = env.reset()
        key = ""
        action = "w"
        while key != "q":
            env.render()
            key = input("action: ")
            if key   == "w": action = UP
            elif key == "d": action = RIGHT
            elif key == "s": action = DOWN
            elif key == "a": action = LEFT
            elif key == "f": action = 5
            else: action = STAY
            obs, rew, done, info = env.step(action)
            print("rew:", rew)
            print("done:", done)
            print("info")
            for k in info.keys():
                print("    ", k, ":", info[k])
            print("Immediate IsPop:", env.controller.is_pop())
            print("Immediate prevskip:", env.controller.prev_skipped)
            print("Immediate skip:", env.controller.skipped)
            print("Seed:", args["seed"])
            if done:
                obs = env.reset()
                print("Immediate IsPop:", 1)
                print("Immediate prevskip:", 0)
                print("Immediate skip:", 0)
