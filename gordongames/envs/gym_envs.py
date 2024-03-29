import os, subprocess, time, signal
import gym
import gym.spaces as spaces
#from gordongames.envs.ggames import Discrete
from gordongames.envs.ggames.controllers import *
from gordongames.envs.ggames.constants import GRAB, STAY, ITEM, TARG, PLAYER, PILE, BUTTON, OBJECT_TYPES
from gordongames.envs.ggames.utils import find_empty_space_along_row
import numpy as np
import time

try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError as e:
    raise OSError.DependencyNotInstalled("{}. (HINT: see matplotlib documentation for installation https://matplotlib.org/faq/installing_faq.html#installation".format(e))

class GordonGame(gym.Env):
    """
    The base class for all gordongames variants.

    NOTE: in general, the value for all info values is shifted one step
    so that it represents the number of items for the observation of the
    previous time step.
    This is because when processing the data, we usually have one more
    image observation than game infos (due to the env.reset() function not
    returning an info dict until recently). We then ignore the final
    observation of the game because it's unimportant to the model.
    value for is_pop (is_player_on_pile) represents the current time
    step.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 targ_range=(1,10),
                 grid_size=(31,31),
                 pixel_density=5,
                 harsh=True,
                 max_steps=None,
                 hold_outs=set(),
                 rand_pdb=True,
                 sym_distr=True,
                 player_on_pile=True,
                 rand_timing=False,
                 timing_p=0.8,
                 spacing_limit=None,
                 zipf_exponent=None,
                 min_play_area=False,
                 n_held_outs=0,
                 center_signal=True,
                 *args, **kwargs):
        """
        Args:
            targ_range: tuple of ints (low, high) (both inclusive)
                the range of potential target counts for the game
            grid_size: tuple of ints (n_row, n_col)
                the dimensions of the grid in grid units
            pixel_density: int
                the number of pixels per unit in the grid
            harsh: bool
                changes the reward system to be more continuous if false
            max_steps: positive valued int or None
                the maximum number of steps an episode can take.
            hold_outs: set of ints
                a set of integer values representing numbers of targets
                that should not be sampled when sampling targets
            rand_pdb: bool
                if true, the player, dispenser, and button are randomly
                placed along the topmost row at the beginning of the
                game. Otherwise they are placed
            sym_distr: bool
                if false and rand_pdb is false, the player, dispenser,
                and button are consistently distributed the same way
                at initialization on every episode. Otherwise the initial
                distribution is reflected about the yaxis with 50% prob.
                Only applies when rand_pdb is false.
            player_on_pile: bool
                if true, the player always starts on top of the dispenser
                pile in counting games. If false, it may or may not.
            rand_timing: bool
                if true, the number of frames after each pixel reveal
                is uniformly selected from 1-2 frames. This forces the
                numeric signal to be derived from the number of pixels
                rather than the number of frames.
            timing_p: float between 0 and 1
                the probability of displaying the next target item. the
                animation phase continues until all targets are
                displayed but with probability 1-timing_p no new targets
                are displayed for any given step in the animation phase.
                This is used to discourage the model from counting the
                number of frames rather than the items.
            spacing_limit: None or int greater than 0
                if greater than 0, limits the spacing between the
                player, dispenser, and ending button to be within
                spacing_limit steps on either side of the dispenser's
                initial position. If rand_locs is false, the ending
                button will always be spacing_limit steps away
            zipf_exponent: float or None
                if not None, the target quantities are sampled
                proportionally to the zipfian distribution with the
                argued exponent. p = 1/(n^z) where n is the target
                quantity, z is the zipfian exponent and p is the
                likelihood.
            min_play_area: bool
                if true, minimizes the play area (area above the
                dividing line of the grid) to 4 rows. Otherwise,
                dividing line is placed at approximately the middle
                row of the grid.
            n_held_outs: int
                the number of held out coordinates per target quantity
            center_signal: bool
                if true, signal coord will be centered in demonstration
                area. Otherwise a signal pixel will appear on both
                edges of the grid one row down from the top.
        """
        # determines the unit dimensions of the grid
        self.grid_size = grid_size
        # determines the number of pixels per grid unit
        self.pixel_density = pixel_density
        # tracks number of steps in episode
        self.step_count = 0
        # used in calculations of self.max_steps
        self.max_step_base = self.grid_size[0]//2*self.grid_size[1]*2
        # gets set in reset(), limits number of steps per episode
        self.master_max_steps = max_steps
        self.max_steps = max_steps
        self.targ_range = targ_range
        if type(targ_range) == int:
            self.targ_range = (targ_range,targ_range)
        self.harsh = harsh
        if hold_outs is None: hold_outs = set()
        self.hold_outs = set(hold_outs)
        self.rand_pdb = rand_pdb
        self.sym_distr = sym_distr
        self.player_on_pile = player_on_pile
        self.rand_timing = rand_timing
        self.timing_p = timing_p
        self.spacing_limit = spacing_limit
        self.zipf_exponent = zipf_exponent
        self.min_play_area = min_play_area
        self.n_held_outs = n_held_outs
        if n_held_outs is None: self.n_held_outs = 0
        self.center_signal = center_signal
        self.viewer = None
        self.action_space = spaces.Discrete(6)
        self.is_grabbing = False
        self.seed(int(time.time()))
        self.set_controller()
        obs = np.zeros(
            [g*self.pixel_density for g in self.grid_size],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=obs+np.min(list(COLORS.values())),
            high=obs+np.max(list(COLORS.values()))
        )

    def set_controller(self, contr_kwargs=None):
        """
        Must override this function and set a member `self.controller`
        """
        if not hasattr(self, "controller_type"):
            raise NotImplemented
        if contr_kwargs is None:
            contr_kwargs = {
                "grid_size": self.grid_size,
                "pixel_density": self.pixel_density,
                "harsh": self.harsh,
                "targ_range": self.targ_range,
                "zipf_exponent": self.zipf_exponent,
                "hold_outs": self.hold_outs,
                "rand_pdb": self.rand_pdb,
                "sym_distr": self.sym_distr,
                "player_on_pile": self.player_on_pile,
                "rand_timing": self.rand_timing,
                "timing_p": self.timing_p,
                "spacing_limit": self.spacing_limit,
                "min_play_area": self.min_play_area,
                "n_held_outs": self.n_held_outs,
                "center_signal": self.center_signal,
            }
        self.controller = self.controller_type(**contr_kwargs)
        self.controller.rand = self.rand
        self.controller.reset()

    def _toggle_grab(self):
        """
        Toggles the grab state of the player. If the task is either the
        brief presentation or nuts in can task, the grabbing is
        restricted until the initial animations have finished.
        """
        grab = not self.is_grabbing
        coord = self.controller.register.player.coord
        if self.is_grabbing:
            self.is_grabbing = False
        # we know is_grabbing is currently false and there is an object
        # under the player
        elif not self.controller.register.is_empty(coord):
            self.is_grabbing = True
        # Restrict grabbing if experiencing initial animations in
        # BriefPresentation or NutsInCan tasks.
        if type(self.controller)==BriefPresentationController and\
                self.controller.is_animating:
            self.is_grabbing = False
            grab = False
        return grab

    def step(self, action):
        """
        Args:
            action: int
                the action should be an int of either a direction or
                a grab command
                    0: null action
                    1: move up one unit
                    2: move right one unit
                    3: move down one unit
                    4: move left one unit
                    5: grab/drop object
        Returns:
            last_obs: ndarray
                the observation
            rew: float
                the reward
            done: bool
                if true, the episode has ended
            info: dict
                whatever information the game contains
        """
        self.step_count += 1
        if action != GRAB:
            direction = action
            grab = self.is_grabbing
        else:
            direction = STAY
            grab = self._toggle_grab()
        self.last_obs,rew,done,info = self.controller.step(
            direction,
            int(grab)
        )
        player = self.controller.register.player
        info["grab"] = self.get_other_obj_idx(player, grab)
        info["player_loc"] = self.controller.register.player.coord
        info["count_loc"] = self.controller.register.pile.coord
        info["end_loc"] = self.controller.register.button.coord
        if self.step_count > self.max_steps: done = True
        elif self.step_count == self.max_steps and rew == 0:
            rew = self.controller.max_punishment
            done = True
        return self.last_obs, rew, done, info

    def get_other_obj_idx(self, obj, grab):
        """
        Finds and returns an int representing the first game object
        that is not the argued object and is located at the locations
        of the argued object. The priority of objects is detailed
        by the priority list.

        Args:
            obj: GameObject
            grab: bool
                the player's current grab state
        Returns:
            other_obj: GameObject or None
                one of the other objects located at this location.
                The priority goes by type, see `priority`. a return
                of 0 means the player is either not grabbing or there
                are no items to grab
        """
        # Langpractice depends on this order
        keys = sorted(list(PRIORITY2TYPE.keys()))
        priority = [ PRIORITY2TYPE[k] for k in keys ]
        if not grab: return 0
        reg = self.controller.register.coord_register
        objs = {*reg[obj.coord]}
        if len(objs) == 1: return 0
        objs.remove(obj)
        memo = {o: set() for o in priority}
        # Sort objects
        for o in objs:
            memo[o.type].add(o)
        # return single object by priority
        for o in priority:
            if len(memo[o]) > 0: return TYPE2PRIORITY[o]
        return 0

    def reset_max_steps(self, max_steps=None):
        """
        Needs controller before calling!!

        Args:
            max_steps: positive int or None
                the maximum number of steps per episode
        """
        if max_steps is None or max_steps<=0:
            if self.master_max_steps is None or self.master_max_steps<=0:
                m = (self.controller.n_targs+1)*self.max_step_base
                self.max_steps = m
            else: self.max_steps = self.master_max_steps
        else: self.max_steps = max_steps

    def reset(self, n_targs=None, max_steps=None, held_out=False,
                                                *args, **kwargs):
        self.controller.rand = self.rand
        self.controller.reset(n_targs=n_targs, held_out=held_out)
        self.reset_max_steps(max_steps)
        self.is_grabbing = False
        self.step_count = 0
        self.last_obs = self.controller.grid.grid
        return self.last_obs, {}

    def render(self, mode='human', close=False, frame_speed=.1):
        if self.viewer is None:
            self.fig = plt.figure()
            self.viewer = self.fig.add_subplot(111)
            plt.ion()
            self.fig.show()
        else:
            self.viewer.clear()
            self.viewer.imshow(self.last_obs)
            plt.pause(frame_speed)
        self.fig.canvas.draw()

    def seed(self, x):
        self.rand = np.random.default_rng(x)

    def set_seed(self, x):
        self.rand = np.random.default_rng(x)
        pass

class EvenLineMatch(GordonGame):
    """
    Creates a gym version of Peter Gordon's Even Line Matching game.
    The user attempts to match the target object line within a maximum
    number of steps based on the size of the grid and the number of
    target objects on the grid. The maximum step count is enough so
    that the agent can walk around the perimeter of the playable area
    n_targs+1 number of times. The optimal policy will always be able
    to finish well before this.
    """
    def set_controller(self):
        self.controller_type = EvenLineMatchController
        super().set_controller()

class ClusterMatch(GordonGame):
    """
    Creates a gym version of Peter Gordon's Cluster Matching game.
    The user attempts to place the same number of items on the grid as
    the number of target objects. The target objects are randomly
    placed while the agent attempts to align the placed items along a
    single row.

    The number of steps is based on the size of the grid and the number
    of target objects on the grid. The maximum step count is enough so
    that the agent can walk around the perimeter of the playable area
    n_targs+1 number of times. The optimal policy will always be able
    to finish well before this.
    """
    def set_controller(self):
        self.controller_type = ClusterMatchController
        super().set_controller()

class OrthogonalLineMatch(GordonGame):
    """
    Creates a gym version of Peter Gordon's Orthogonal Line Matching
    game.  The user attempts to layout a number of items in a horizontal
    line to match the count of a number of target objects laying in a
    vertical line. It must do so within a maximum number of steps
    based on the size of the grid and the number of target objects on
    the grid. The maximum step count is enough so that the agent can
    walk around the perimeter of the playable area n_targs+1 number of
    times. The optimal policy will always be able to finish well
    before this.
    """
    def set_controller(self):
        self.controller_type = OrthogonalLineMatchController
        super().set_controller()

class UnevenLineMatch(GordonGame):
    """
    Creates a gym version of Peter Gordon's Uneven Line Matching game.
    The user attempts to match the target object line within a maximum
    number of steps based on the size of the grid and the number of
    target objects on the grid. The maximum step count is enough so
    that the agent can walk around the perimeter of the playable area
    n_targs+1 number of times. The optimal policy will always be able
    to finish well before this.
    """
    def set_controller(self):
        self.controller_type = UnevenLineMatchController
        super().set_controller()

class ReverseClusterMatch(GordonGame):
    """
    Creates a gym version of the reverse of Peter Gordon's Cluster
    Matching game. The user attempts to place the same number of items
    on the grid as the number of evenly spaced, aligned target objects.
    The placed items must not align with the target objects.

    The number of steps is based on the size of the grid and the number
    of target objects on the grid. The maximum step count is enough so
    that the agent can walk around the perimeter of the playable area
    n_targs+1 number of times. The optimal policy will always be able
    to finish well before this.
    """
    def set_controller(self):
        self.controller_type = ReverseClusterMatchController
        super().set_controller()

class ClusterClusterMatch(GordonGame):
    """
    Creates a gym game in which the user attempts to place the same
    number of items on the grid as the number of target objects.
    The target objects are randomly placed and no structure is imposed
    on the placement of the user's items.

    The number of steps is based on the size of the grid and the number
    of target objects on the grid. The maximum step count is enough so
    that the agent can walk around the perimeter of the playable area
    n_targs+1 number of times. The optimal policy will always be able
    to finish well before this.
    """
    def set_controller(self):
        self.controller_type = ClusterClusterMatchController
        super().set_controller()

class BriefPresentation(GordonGame):
    """
    Creates a gym game in which the user attempts to place the same
    number of items on the grid as the number of target objects.
    The target objects are randomly placed and the agent is supposed
    to place the same number of items aligned along a single row. The
    agent's movement is restricted for the first DISPLAY_COUNT frames.
    The targets are removed from the agent's visual display after the
    DISPLAY_COUNT frames and the agent has to perform the counting task
    from memory.

    The number of steps is based on the size of the grid and the number
    of target objects on the grid. The maximum step count is enough so
    that the agent can walk around the perimeter of the playable area
    n_targs+1 number of times. The optimal policy will always be able
    to finish well before this.
    """
    def set_controller(self):
        self.controller_type = BriefPresentationController
        super().set_controller()

class NavigationTask(GordonGame):
    """
    Creates a gym version of a simple navigation task.

    This class creates a game in which the environment initializes
    with items, player, button, and dispenser all in the playable
    half of the grid. The player must then navigate to all items
    and drag them back to the dispenser. Then they must end the game
    using the ending button.
    """
    def set_controller(self):
        self.controller_type = NavigationTaskController
        super().set_controller()

    def step(self, action):
        """
        Args:
            action: int
                the action should be an int of either a direction or
                a grab command
                    0: null action
                    1: move up one unit
                    2: move right one unit
                    3: move down one unit
                    4: move left one unit
                    5: grab/drop object
        Returns:
            last_obs: ndarray
                the observation
            rew: float
                the reward
            done: bool
                if true, the episode has ended
            info: dict
                whatever information the game contains
        """
        self.last_obs, rew, done, info = super().step(action)
        info["grab"] = done or info["grab"]
        info["player_loc"] = self.controller.register.player.coord
        info["count_loc"] = self.controller.register.pile.coord
        info["end_loc"] = self.controller.register.button.coord
        return self.last_obs, rew, done, info

class CanTask(GordonGame):
    """
    This is an abstract class to unify some code between NutsInCan
    varieties.
    """
    def step(self, action):
        """
        Args:
            action: int
                the action should be an int of either a direction or
                a grab command
                    0: null action
                    1: move up one unit
                    2: move right one unit
                    3: move down one unit
                    4: move left one unit
                    5: grab/drop object
        Returns:
            last_obs: ndarray
                the observation
            rew: float
                the reward
            done: bool
                if true, the episode has ended
            info: dict
                whatever information the game contains
        """
        self.step_count += 1

        direction = STAY
        grab = 0
        if action < 5:
            direction = action
            grab = 0 # We always handle grabs here
        elif not self.controller.is_animating:
            coord = self.controller.register.player.coord
            if not self.controller.register.is_empty(coord):
                grab = 1

        # Check if player grabbed the pile
        grabbed_type = self.get_other_obj_idx(
            self.controller.register.player,
            grab
        )
        # Other option is if it's an item, but this will only happen
        # when the agent grabs one that has been placed by the env.
        if grabbed_type == TYPE2PRIORITY[PILE]:
            self.place_item()
            direction = STAY
            grab = 0
        elif grabbed_type == TYPE2PRIORITY[BUTTON]:
            direction = STAY
            grab = 1

        self.last_obs,rew,done,info = self.controller.step(
            direction,
            grab
        )
        player = self.controller.register.player
        if self.step_count > self.max_steps: done = True
        elif self.step_count == self.max_steps and rew == 0:
            rew = self.controller.max_punishment
            done = True
        info["grab"] = done or grab or grabbed_type==TYPE2PRIORITY[PILE]
        info["player_loc"] = self.controller.register.player.coord
        info["count_loc"] = self.controller.register.pile.coord
        info["end_loc"] = self.controller.register.button.coord
        if grabbed_type==TYPE2PRIORITY[PILE]: info["n_items"] -= 1
        return self.last_obs, rew, done, info

    def place_item(self):
        """
        Creates and places an item along a row.
        """
        player = self.controller.register.player
        middle = self.controller.register.grid.middle_row
        row = 2
        coord = None
        while row < middle and coord is None:
            coord = find_empty_space_along_row(
                self.controller.register,
                (row, player.coord[1])
            )
            row += 1
        if coord is None: return
        self.controller.register.make_object(
            obj_type=ITEM,
            coord=coord
        )

class NutsInCan(CanTask):
    """
    Creates a gym version of Peter Gordon's Nuts-In-A-Can game.

    This class creates a game in which the environment initially flashes
    the targets one by one until all targets are flashed. At the end
    of the flashing, a center piece appears (to indicate that the
    flashing stage is over). The agent must then grab the pile the same
    number of times as there are targets (each of which was flashed only
    briefly at the beginning of the game).

    Items corresponding to the number of pile grabs by the agent will
    automatically align themselves in a neat row after each pile grab.
    Once the agent believes the number of items is equal to the number
    of targets, they must press the ending button.

    If the agent exceeds the number of targets, the items will continue
    to display until the total quantity of items doubles that of the
    targets.

    The number of steps is based on the size of the grid and the number
    of target objects on the grid. The maximum step count is enough so
    that the agent can walk around the perimeter of the playable area
    n_targs+1 number of times. The optimal policy will always be able
    to finish well before this.
    """
    def set_controller(self):
        self.controller_type = NutsInCanController
        super().set_controller()

class VisNuts(CanTask):
    """
    Creates a gym version of Peter Gordon's Nuts-In-A-Can game in which
    the nuts remain visible.

    This class creates a game in which the environment has an initial
    animation in which the targets are flashed one by one until all
    targets are visible. At the end of the animation, a center piece
    appears (as an indication that the flashing stage is over).
    The agent must then grab the pile the same
    number of times as there are targets.

    Items corresponding to the number of pile grabs by the agent will
    automatically align themselves in a neat row after each pile grab.
    Once the agent believes the number of items is equal to the number
    of targets, they must press the ending button.

    If the agent exceeds the number of targets, the items will continue
    to display until the total quantity of items doubles that of the
    targets.

    The number of steps is based on the size of the grid and the number
    of target objects on the grid. The maximum step count is enough so
    that the agent can walk around the perimeter of the playable area
    n_targs+1 number of times. The optimal policy will always be able
    to finish well before this.
    """
    def set_controller(self):
        self.controller_type = VisNutsController
        super().set_controller()

class StaticVisNuts(CanTask):
    """
    Creates a gym version of Peter Gordon's Nuts-In-A-Can game in which
    the nuts remain visible. The difference between this variant and
    the VisNuts variant is that in this variant, all nuts are displayed
    immediately at the beginning. The agent is still unable to interact
    with the button until the requisite number of frames has passed.

    This class creates a game in which the environment has an initial
    animation in which the targets are all displayed at the very
    beginning. The same number of frames as there are targets passes
    at which point a center signal piece
    appears (as an indication that the initial animation stage is over).
    The agent must then grab the pile the same
    number of times as there are targets.

    Items corresponding to the number of pile grabs by the agent will
    automatically align themselves in a neat row after each pile grab.
    Once the agent believes the number of items is equal to the number
    of targets, they must press the ending button.

    If the agent exceeds the number of targets, the items will continue
    to display until the total quantity of items doubles that of the
    targets.

    The number of steps is based on the size of the grid and the number
    of target objects on the grid. The maximum step count is enough so
    that the agent can walk around the perimeter of the playable area
    n_targs+1 number of times. The optimal policy will always be able
    to finish well before this.
    """
    def set_controller(self):
        self.controller_type = StaticVisNutsController
        super().set_controller()

class GiveN(GordonGame):
    """
    An abstract class to unify code for the visible and invisible give
    n task variants.
    """
    def step(self, action):
        """
        Args:
            action: int
                the action should be an int of either a direction or
                a grab command
                    0: null action
                    1: move up one unit
                    2: move right one unit
                    3: move down one unit
                    4: move left one unit
                    5: grab/drop object
        Returns:
            last_obs: ndarray
                the observation
            rew: float
                the reward
            done: bool
                if true, the episode has ended
            info: dict
                whatever information the game contains
        """
        self.step_count += 1

        direction = STAY
        grab = 0
        if action < 5:
            direction = action
            grab = 0 # We always handle grabs here
        elif not self.controller.is_animating:
            coord = self.controller.register.player.coord
            if not self.controller.register.is_empty(coord):
                grab = 1

        # Check if player grabbed the pile
        grabbed_type = self.get_other_obj_idx(
            self.controller.register.player,
            grab
        )
        # Other option is if it's an item, but this will only happen
        # when the agent grabs one that has been placed by the env.
        if grabbed_type == TYPE2PRIORITY[PILE]:
            self.place_item()
            direction = STAY
            grab = 0
        elif grabbed_type == TYPE2PRIORITY[BUTTON]:
            direction = STAY
            grab = 1

        self.last_obs,rew,done,info = self.controller.step(
            direction,
            grab
        )
        player = self.controller.register.player
        if self.step_count > self.max_steps: done = True
        elif self.step_count == self.max_steps and rew == 0:
            rew = self.controller.max_punishment
            done = True
        info["grab"] = done or grab or grabbed_type==TYPE2PRIORITY[PILE]
        info["player_loc"] = self.controller.register.player.coord
        info["count_loc"] = self.controller.register.pile.coord
        info["end_loc"] = self.controller.register.button.coord
        if grabbed_type==TYPE2PRIORITY[PILE]: info["n_items"] -= 1
        return self.last_obs, rew, done, info

    def place_item(self):
        """
        Creates and places an item along a row.
        """
        player = self.controller.register.player
        middle = self.controller.register.grid.middle_row
        row = 2
        coord = None
        while row < middle and coord is None:
            coord = find_empty_space_along_row(
                self.controller.register,
                (row, player.coord[1])
            )
            row += 1
        if coord is None: return
        self.controller.register.make_object(
            obj_type=ITEM,
            coord=coord
        )

class InvisN(GiveN):
    """
    Creates a gym version of a give n task with no visual numeric signal.
    Due to the interface of
    the game, this just means that the target quantity is not displayed
    at any point in the episode. Intead the user must use the n_targs
    value in the info dict.
    """
    def set_controller(self):
        self.controller_type = InvisNController
        super().set_controller()

class VisN(GiveN):
    """
    Creates a gym version of a give n task in which n target items are
    visible from the beginning. The target items are randomly distributed
    in the demonstration area. For any comparisons to the Give N task,
    it probably makes sense for the user to use the n_targs value in
    the info dict as an additional language signal for the policy network.
    """
    def set_controller(self):
        self.controller_type = VisNController
        super().set_controller()

