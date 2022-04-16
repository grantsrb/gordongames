import gordoncont.ggames.controllers as controllers
import matplotlib.pyplot as plt
from gordoncont.ggames.ai import even_line_match, cluster_match, rev_cluster_match
from gordoncont.ggames.constants import DIRECTION2STR

if __name__=="__main__":
    targ_range = (1,13)
    grid_size = (15,15)
    pixel_density=4
    harsh=True
    contrs = [
        controllers.EvenLineMatchController,
        controllers.ClusterMatchController,
        controllers.ReverseClusterMatchController,
    ]
    ais = [
        even_line_match,
        cluster_match,
        rev_cluster_match
    ]
    for c,ai in zip(contrs,ais):
        contr = c(
            targ_range=targ_range,
            grid_size=grid_size,
            pixel_density=pixel_density,
            harsh=harsh
        )
        for i in range(2):
            contr.harsh = not contr.harsh
            obs = contr.reset()
            done = False
            count = 0
            while not done:
                print("looping", count)
                count += 1
                xy, grab = ai(contr)
                print("xy:", xy)
                print("Coord:", contr.register.xy2coord(xy))
                print("grab:", grab)
                obs, rew, done, info = contr.step(xy,grab)
                print("done: ", done)
                print("rew: ", rew)
                plt.imshow(obs)
                plt.show()

