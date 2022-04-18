import gordongames as gg
import numpy as np

class Oracle:
    def __call__(self, env=None, state=None):
        """
        All oracles must implement this function to operate on the
        environment.

        Args:
            env: None or SequentialEnvironment
                the environment to be acted upon. if None, state must
                be not None
            state: None or torch FloatTensor
                the environment to be acted upon. if None, env must
                be not None.
        """
        raise NotImplemented

class NullOracle(Oracle):
    def __call__(self, *args, **kwargs):
        return 0

class RandOracle(Oracle):
    def __init__(self, actn_min=0, actn_max=5):
        self.brain = lambda: np.random.randint(actn_min, actn_max)

    def __call__(self, *args, **kwargs):
        return self.brain()

class GordonOracle(Oracle):
    def __init__(self, env_type, *args, **kwargs):
        self.env_type = env_type
        self.is_grabbing = False
        
        if self.env_type == "gordongames-v0":
            self.brain = gg.envs.ggames.ai.even_line_match
        elif self.env_type == "gordongames-v1":
            self.brain = gg.envs.ggames.ai.cluster_match
        elif self.env_type == "gordongames-v2":
            self.brain = gg.envs.ggames.ai.cluster_match
        elif self.env_type == "gordongames-v3":
            self.brain = gg.envs.ggames.ai.even_line_match
        elif self.env_type == "gordongames-v4":
            self.brain = gg.envs.ggames.ai.nuts_in_can
        elif self.env_type == "gordongames-v5":
            self.brain = gg.envs.ggames.ai.rev_cluster_match
        elif self.env_type == "gordongames-v6":
            self.brain = gg.envs.ggames.ai.rev_cluster_match
        elif self.env_type == "gordongames-v7":
            self.brain = gg.envs.ggames.ai.brief_display
        elif self.env_type == "gordongames-v8":
            self.brain = gg.envs.ggames.ai.nuts_in_can
        else:
            raise NotImplemented

    def __call__(self, env, *args, **kwargs):
        """
        Args:
            env: SequentialEnvironment
                the environment
        """
        (direction, grab) = self.brain(env.controller)
        if grab == env.is_grabbing:
            return direction
        elif self.brain == gg.envs.ggames.ai.nuts_in_can:
            return gg.envs.ggames.constants.GRAB
        else:
            return gg.envs.ggames.constants.GRAB
