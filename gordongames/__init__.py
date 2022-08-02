import gordongames.oracles
import gordongames.envs
from gym.envs.registration import register

register(
    id='gordongames-v0',
    entry_point='gordongames.envs:EvenLineMatch',
)
register(
    id='gordongames-v1',
    entry_point='gordongames.envs:ClusterMatch',
)
register(
    id='gordongames-v2',
    entry_point='gordongames.envs:OrthogonalLineMatch',
)
register(
    id='gordongames-v3',
    entry_point='gordongames.envs:UnevenLineMatch',
)
register(
    id='gordongames-v4',
    entry_point='gordongames.envs:NutsInCan',
)
register(
    id='gordongames-v5',
    entry_point='gordongames.envs:ReverseClusterMatch',
)
register(
    id='gordongames-v6',
    entry_point='gordongames.envs:ClusterClusterMatch',
)
register(
    id='gordongames-v7',
    entry_point='gordongames.envs:BriefPresentation',
)
register(
    id='gordongames-v8',
    entry_point='gordongames.envs:VisNuts',
)
register(
    id='gordongames-v9',
    entry_point='gordongames.envs:NavigationTask',
)
register(
    id='gordongames-v10',
    entry_point='gordongames.envs:StaticVisNuts',
)
