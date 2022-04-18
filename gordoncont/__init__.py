import gordoncont.oracles
import gordoncont.envs
import gordoncont.ggames
from gym.envs.registration import register

register(
    id='gordoncont-v0',
    entry_point='gordoncont.envs:EvenLineMatch',
)
register(
    id='gordoncont-v1',
    entry_point='gordoncont.envs:ClusterMatch',
)
register(
    id='gordoncont-v2',
    entry_point='gordoncont.envs:OrthogonalLineMatch',
)
register(
    id='gordoncont-v3',
    entry_point='gordoncont.envs:UnevenLineMatch',
)
register(
    id='gordoncont-v4',
    entry_point='gordoncont.envs:NutsInCan',
)
register(
    id='gordoncont-v5',
    entry_point='gordoncont.envs:ReverseClusterMatch',
)
register(
    id='gordoncont-v6',
    entry_point='gordoncont.envs:ClusterClusterMatch',
)
register(
    id='gordoncont-v7',
    entry_point='gordoncont.envs:BriefPresentation',
)
register(
    id='gordoncont-v8',
    entry_point='gordoncont.envs:VisNuts',
)
