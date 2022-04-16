import gordoncont as gg
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
        return [0,0,-1]

class RandOracle(Oracle):
    def __init__(self, actn_min=0, actn_max=5):
        self.brain = lambda: (np.random.random((3,))-.5)/0.5

    def __call__(self, *args, **kwargs):
        actn = self.brain()
        grab = actn[2]>=0
        temp = 0.05
        return [*actn[:2], (grab-.5)/(.5+temp)]

class GordonOracle(Oracle):
    def __init__(self, env_type, *args, **kwargs):
        self.env_type = env_type
        self.is_grabbing = False

        if self.env_type ==   "gordoncont-v0":
            self.brain = gg.ggames.ai.even_line_match
        elif self.env_type == "gordoncont-v1":
            self.brain = gg.ggames.ai.cluster_match
        elif self.env_type == "gordoncont-v2":
            self.brain = gg.ggames.ai.cluster_match
        elif self.env_type == "gordoncont-v3":
            self.brain = gg.ggames.ai.even_line_match
        elif self.env_type == "gordoncont-v4":
            self.brain = gg.ggames.ai.nuts_in_can
        elif self.env_type == "gordoncont-v5":
            self.brain = gg.ggames.ai.rev_cluster_match
        elif self.env_type == "gordoncont-v6":
            self.brain = gg.ggames.ai.rev_cluster_match
        elif self.env_type == "gordoncont-v7":
            self.brain = gg.ggames.ai.brief_display
        elif self.env_type == "gordoncont-v8":
            self.brain = gg.ggames.ai.nuts_in_can
        else:
            raise NotImplemented

    def __call__(self, env, *args, **kwargs):
        """
        Args:
            env: SequentialEnvironment
                the environment
        """
        (xycoord, grab) = self.brain(env.controller)
        # use a temperature parameter to avoid vanishing gradients
        temp = .05
        actn = [*xycoord, (float(grab)-.5)/(.5+temp)]
        if grab == env.is_grabbing:
            return actn
        elif self.brain == gg.ggames.ai.nuts_in_can:
            actn = [*xycoord, .5/(.5+temp)]
            return actn
        else:
            actn = [*xycoord, .5/(.5+temp)]
            return actn

