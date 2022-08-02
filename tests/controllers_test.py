import gordongames.envs.ggames.controllers as controllers
import matplotlib.pyplot as plt
from gordongames.envs.ggames.ai import even_line_match, cluster_match
from gordongames.envs.ggames.constants import DIRECTION2STR

if __name__=="__main__":
    targ_range = (1,13)
    grid_size = (15,15)
    pixel_density=1
    harsh=True
    contr = controllers.ClusterMatchController(
        targ_range=targ_range,
        grid_size=grid_size,
        pixel_density=pixel_density,
        harsh=harsh
    )
    for i in range(10):
        contr.harsh = not contr.harsh
        obs = contr.reset()
        done = False
        count = 0
        while not done:
            print("looping", count)
            count += 1
            direction, grab = cluster_match(contr)
            print("direction:", DIRECTION2STR[direction])
            print("grab:", grab)
            obs, rew, done, info = contr.step(direction, grab)
            print("done: ", done)
            print("rew: ", rew)
            plt.imshow(obs)
            plt.show()

