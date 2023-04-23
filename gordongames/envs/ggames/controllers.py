from gordongames.envs.ggames.grid import Grid
from gordongames.envs.ggames.registry import Register
from gordongames.envs.ggames.constants import *
from gordongames.envs.ggames.utils import get_rows_and_cols, get_aligned_items, get_max_row, zipfian
import numpy as np
import time

"""
This file contains each of the game controller classes for each of the
Gordon games. 
"""

class Controller:
    """
    The base controller class for handling initializations. It is
    abstract and as such should not be implemented directly. It should
    handle all game logic by manipulating the register.
    """
    def __init__(self,
                 targ_range: tuple=(1,10),
                 grid_size: tuple=(31,31),
                 pixel_density: int=1,
                 hold_outs: set=set(),
                 rand_pdb: bool=True,
                 sym_distr: bool=True,
                 player_on_pile: bool=False,
                 spacing_limit=None,
                 rand_timing: bool=False,
                 timing_p: float=0.8,
                 zipf_exponent=None,
                 min_play_area=False,
                 n_held_outs=0,
                 center_signal=True,
                 *args, **kwargs):
        """
        targ_range: tuple (Low, High) (inclusive)
            the low and high number of targets for the game. each
            episode is intialized with a number of targets within
            this range.
        grid_size: tuple (Row, Col)
            the dimensions of the grid in grid units
        pixel_density: int
            the side length of a single grid unit in pixels
        hold_outs: set of ints
            a set of integer values representing numbers of targets
            that should not be sampled when sampling targets
        player_on_pile: bool
            if true, the player always starts on top of the dispenser
            pile in counting games. If false, it may or may not.
        spacing_limit: None or int greater than 0
            if greater than 0, limits the spacing between the
            player, dispenser, and ending button to be within
            spacing_limit steps on either side of the dispenser's
            initial position. If rand_locs is false, the ending
            button will always be spacing_limit steps away
        rand_pdb: bool
            if true, the player, dispenser, and button are randomly
            placed along the topmost row at the beginning of each
            episode. Otherwise, they are placed evenly spaced in the
            order player, dispenser, button from left to right.
        sym_distr: bool
            if false and rand_pdb is false, the player, dispenser,
            and button are consistently distributed the same way
            at initialization on every episode. Otherwise the initial
            distribution is reflected about the yaxis with 50% prob.
            Only applies when rand_pdb is false.
        rand_timing: bool
            if true, the number of frames for a pixel to be displayed
            at the beginning is uniformly sampled from 1-2
        timing_p: float between 0 and 1
            the probability of displaying the next target item. the
            animation phase continues until all targets are displayed
            but with probability 1-timing_p no new targets are displayed
            for any given step in the animation phase. This is used
            to discourage the model from counting the number of frames
            rather than the items.
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
        if type(targ_range) == int:
            targ_range = (targ_range, targ_range)
        assert targ_range[0] <= targ_range[1]
        assert targ_range[0] >= 0 and targ_range[1] < grid_size[1]
        self._targ_range = targ_range
        self._grid_size = grid_size
        self._pixel_density = pixel_density
        self._hold_outs = set(hold_outs)
        self.rand_pdb = rand_pdb
        self.sym_distr = sym_distr
        self.player_on_pile = player_on_pile
        self.spacing_limit = spacing_limit
        self.zipf_exponent = zipf_exponent
        self.min_play_area = min_play_area
        self.rand_timing = rand_timing
        self.timing_p = timing_p
        trgs = set(range(targ_range[0],targ_range[1]+1))
        assert len(trgs-hold_outs)>0
        self.is_animating = False
        self.rand = np.random.default_rng(int(time.time()))
        self.n_steps = 0
        self.n_held_outs = n_held_outs
        self.center_signal = center_signal

    @property
    def targ_range(self):
        return self._targ_range

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def density(self):
        return self._pixel_density

    @property
    def hold_outs(self):
        return self._hold_outs

    @property
    def n_targs(self):
        return self.register.n_targs

    @property
    def max_punishment(self):
        return -self.targ_range[1]

    def calculate_reward(self):
        raise NotImplemented

    def is_pop(self):
        """
        Function to determine if player is on the pile.
        """
        reg = self.register
        if reg.pile in reg.coord_register[reg.player.coord]:
            return 1
        return 0

    def step(self, direction: int, grab: int):
        """
        Step takes a movement and a grabbing action. The function
        moves the player and any items in the following way.

        If the player was carrying an object and stepped onto another
        object, the game is handled as follows. While the player
        continues to grab, all objects and the player remain overlayn.
        If the player releases the grab button while an object is on
        top of another object, one of 2 things can happen. If the 
        underlying object is a pile, the item is returned to the pile.
        If the underlying object is an item, the previously carried
        item is placed in the nearest empty location in this order.
        Up, right, down, left. If none are available a search algorithm
        is performed spiraling outward from the center. i.e. up 2, up 2
        right 1, up 2 right 2, up 1 right 2, right 2, etc.

        Args:
          direction: int [0, 1, 2, 3, 4]
            Check DIRECTIONS to ensure these values haven't changed
                0: no movement
                1: move up (lower row unit)
                2: move right (higher column unit)
                3: move down (higher row unit)
                4: move left (lower column unit)
          grab: int [0,1]
            grab is an action to enable the agent to carry items around
            the grid. when a player is on top of an item, they can grab
            the item and carry it with them as they move. If the player
            is on top of a pile, a new item is created and carried with
            them to the next square.
        
            0: quit grabbing item
            1: grab item. item will follow player to whichever square
              they move to.
        """
        self.n_steps += 1
        info = {
            "is_harsh": self.harsh,
            "n_targs": self.n_targs,
            "n_items": self.register.n_items,
            "n_aligned": len(get_aligned_items(
                items=self.register.items,
                targs=self.register.targs,
                min_row=0
            )),
            "disp_targs":int(self.register.display_targs),
            "is_animating":int(self.is_animating),
            "is_pop": int(self.is_pop()),
            "skipped": 0,
        }
        if self.n_steps > self.n_targs and self.is_animating:
            self.register.make_signal(center_signal=self.center_signal)
            self.is_animating = False
        if self.n_steps < self.n_targs+1:
            grab = 0
            info["n_items"] = self.n_steps-1
        elif self.n_steps == self.n_targs+1:
            grab = 0
            info["n_items"] = self.n_steps-1

        event = self.register.step(direction, grab)

        done = False
        rew = 0
        if event == BUTTON_PRESS:
            rew = self.calculate_reward(harsh=self.harsh)
            done = True
        elif event == FULL:
            done = True
            rew = -1
        elif event == STEP:
            done = False
            rew = 0
        return self.grid.grid, rew, done, info

    def reset(self, n_targs=None):
        """
        This member must be overridden. Don't forget to reset n_steps!!
        """
        self.n_steps = 0
        raise NotImplemented

class NavigationTaskController(Controller):
    """
    This controller creates an instance of a navigation game. The
    agent must simply navigate to the target items on the grid, bring
    it back to the dispenser, and then end the episode by pressing
    the ending button.
    """
    def __init__(self, harsh=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.harsh = harsh
        self.grid = Grid(
            grid_size=self.grid_size,
            pixel_density=self.density,
            divide=True,
            min_play_area=self.min_play_area
        )
        self.register = Register(self.grid, n_targs=2)

    def init_variables(self, n_targs=None):
        """
        This function should be called everytime the environment starts
        a new episode.
        """
        self.register.rand = self.rand
        self.n_steps = 0
        if n_targs is None: n_targs = 2
        # wipes items from grid and makes/deletes targs
        self.register.reset(n_targs)
        self.is_animating = False
        self.register.display_targs = True

    def reset(self, n_targs=None, *args, **kwargs):
        """
        This function should be called everytime the environment starts
        a new episode.
        """
        self.init_variables(n_targs)
        self.register.navigation_task(
            self.rand_pdb,
            self.player_on_pile,
            self.spacing_limit,
            self.sym_distr
        )
        self.register.make_signal(center_signal=self.center_signal)
        return self.grid.grid

    def calculate_reward(self, harsh: bool=False):
        """
        Determines what reward to return. In this case, checks for the
        number of items on the grid. The goal is 0 items. Returns the
        negative of the number of items. If harsh, returns +1 for 0
        items and -1 otherwise.

        Args:
            harsh: bool
                if true, returns a postive 1 reward only when zero
                items remain on the grid at the end of the episode.
                if false, returns the negative of the number of items
                remaining on the grid at the end of the episode.

                harsh == False:
                    rew = -n_items
                harsh == True:
                    rew = -1*(n_items>0) + (n_items==0)
        Returns:
            rew: float
                the calculated reward
        """
        n_items = self.register.n_items
        if harsh:
            return -1*(n_items>0) + int(n_items==0)
        else:
            return -n_items

    def step(self, direction: int, grab: int):
        """
        Step takes a movement and a grabbing action. The function
        moves the player and any items in the following way.

        If the player was carrying an object and stepped onto another
        object, the game is handled as follows. While the player
        continues to grab, all objects and the player remain overlayn.
        If the player releases the grab button while an object is on
        top of another object, one of 2 things can happen. If the 
        underlying object is a pile, the item is returned to the pile.
        If the underlying object is an item, the previously carried
        item is placed in the nearest empty location in this order.
        Up, right, down, left. If none are available a search algorithm
        is performed spiraling outward from the center. i.e. up 2, up 2
        right 1, up 2 right 2, up 1 right 2, right 2, etc.

        Args:
          direction: int [0, 1, 2, 3, 4]
            Check DIRECTIONS to ensure these values haven't changed
                0: no movement
                1: move up (lower row unit)
                2: move right (higher column unit)
                3: move down (higher row unit)
                4: move left (lower column unit)
          grab: int [0,1]
            grab is an action to enable the agent to carry items around
            the grid. when a player is on top of an item, they can grab
            the item and carry it with them as they move. If the player
            is on top of a pile, a new item is created and carried with
            them to the next square.
        
            0: quit grabbing item
            1: grab item. item will follow player to whichever square
              they move to.
        """
        self.n_steps += 1
        info = {
            "is_harsh": self.harsh,
            "n_targs": self.n_targs,
            "n_items": self.register.n_items,
            "n_aligned": len(get_aligned_items(
                items=self.register.items,
                targs=self.register.targs,
                min_row=0
            )),
            "disp_targs":int(self.register.display_targs),
            "is_animating":int(self.is_animating),
            "is_pop": int(self.is_pop()),
            "skipped": 0,
        }

        event = self.register.step(direction, grab)

        done = False
        rew = 0
        if event == BUTTON_PRESS:
            rew = self.calculate_reward(harsh=self.harsh)
            done = True
        elif event == FULL:
            done = True
            rew = -1
        elif event == STEP:
            done = False
            rew = 0
        return self.grid.grid, rew, done, info

class EvenLineMatchController(Controller):
    """
    This class creates an instance of an Even Line Match game.

    The agent must align a single item along the column of each of the
    target objects.
    """
    def __init__(self, harsh: bool=False, *args, **kwargs):
        """
        See base Controller class for details into arguments.
        
        Args:
            harsh: bool
                if true, returns a postive 1 reward only upon successful
                completion of an episode. if false, returns the
                number of correct target columns divided by the total
                number of columns minus the total number of
                incorrect columns divided by the total number of
                columns.

                harsh == False:
                    rew = n_correct/n_total - n_incorrect/n_total
                harsh == True:
                    rew = n_correct == n_targs
        """
        super().__init__(*args, **kwargs)
        self.grid = Grid(
            grid_size=self.grid_size,
            pixel_density=self.density,
            divide=True,
            min_play_area=self.min_play_area
        )
        self.register = Register(
            self.grid, n_targs=1, n_held_outs=self.n_held_outs
        )
        self.harsh = harsh

    def init_variables(self, n_targs=None):
        """
        This function should be called everytime the environment starts
        a new episode. The animation simply allows a number of frames
        for the agent to count the targets on the grid.
        """
        self.register.rand = self.rand
        self.n_steps = 0
        if n_targs is None:
            low, high = self.targ_range
            sampler = lambda: self.rand.integers(low,high+1)
            if self.zipf_exponent is not None:
                sampler = lambda: zipfian(
                    low, high, self.zipf_exponent, rand=self.rand
                )
            n_targs = sampler()
            while n_targs in self.hold_outs: n_targs = sampler()
        elif n_targs in self.hold_outs:
            print("Overriding holds outs using", n_targs, "targs")
        # wipes items from grid and makes/deletes targs
        self.register.reset(n_targs)
        self.is_animating = True

    def reset(self, n_targs=None, *args, **kwargs):
        """
        This function should be called everytime the environment starts
        a new episode.
        """
        self.init_variables(n_targs)
        self.register.even_line_match(
            rand_pdb=self.rand_pdb,
            player_on_pile=self.player_on_pile,
            spacing_limit=self.spacing_limit,
            sym_distr=self.sym_distr
        )
        return self.grid.grid

    def calculate_reward(self, harsh: bool=False):
        """
        Determines what reward to return. In this case, checks if
        the same number of items exists as targets and checks that
        all items are in a single row and that an item is in each
        column that contains a target. If all of these factors are
        met, if harsh is true, the function returns a reward of 1.
        If harsh is false, the function returns a partial reward
        based on the portion of columns that were successfully filled
        minus the portion of incorrect columns.

        Args:
            harsh: bool
                if true, returns a postive 1 reward only upon successful
                completion of an episode. if false, returns the
                number of correct target columns divided by the total
                number of columns minus the total number of
                incorrect columns divided by the total number of
                columns.

                harsh == False:
                    rew = n_correct/n_total - n_incorrect/n_total
                harsh == True:
                    rew = n_correct == n_targs
        Returns:
            rew: float
                the calculated reward
        """
        targs = self.register.targs
        items = self.register.items
        if harsh and len(targs) != len(items): return -1

        item_rows, item_cols = get_rows_and_cols(items)
        _, targ_cols = get_rows_and_cols(targs)

        if len(item_rows) > 1: return -1
        if harsh:
            if targ_cols == item_cols: return 1
            return -1
        else:
            intersection = targ_cols.intersection(item_cols)
            rew = len(intersection)
            rew -= (len(item_cols)-len(intersection))
            rew -= max(0, np.abs(len(items)-len(targs)))
            return rew

class ClusterMatchController(EvenLineMatchController):
    """
    This class creates an instance of the Cluster Line Match game.

    The agent must place the same number of items as targets along a
    single row. The targets are randomly distributed about the grid.
    """
    def reset(self, n_targs=None, held_out=False):
        """
        This function should be called everytime the environment starts
        a new episode. The animation simply allows a number of frames
        for the agent to count the targets on the grid.

        Args:
            n_targs: int or None
                if int is argued, this will dictate the number of
                target items for the episode
            held_out: bool
                if true, will sample an episode that was held out from
                the non-held out episodes
        """
        self.init_variables(n_targs)
        self.register.cluster_match(
            rand_pdb=self.rand_pdb,
            player_on_pile=self.player_on_pile,
            spacing_limit=self.spacing_limit,
            sym_distr=self.sym_distr,
            held_out=held_out
        )
        return self.grid.grid

    def calculate_reward(self, harsh: bool=False):
        """
        Determines what reward to return. In this case, checks if
        the same number of items exists as targets and checks that
        all items are along a single row.
        
        Args:
            harsh: bool
                if true, the function returns a reward of 1 when the
                same number of items exists as targs and the items are
                aligned along a single row. A -1 is returned otherwise.
                If harsh is false, the function returns a partial
                reward based on the number of aligned items minus the
                number of items over the target count.

                harsh == False:
                    rew = (n_targ - abs(n_items-n_targs))/n_targs
                    rew -= abs(n_aligned_items-n_items)/n_targs
                harsh == True:
                    rew = +1 when n_items == n_targs
                          and
                          n_aligned == n_targs
                    rew = 0 when n_items == n_targs
                          and
                          n_aligned != n_targs
                    rew = -1 otherwise
        Returns:
            rew: float
                the calculated reward
        """
        targs = self.register.targs
        items = self.register.items
        max_row, n_aligned = get_max_row(items,min_row=1,ret_count=True)
        n_targs = len(targs)
        n_items = len(items)
        if harsh:
            if n_items == n_targs: return int(n_aligned == n_targs)
            else: return -1
        else:
            rew = (n_targs - np.abs(n_items-n_targs))/n_targs
            rew -= np.abs(n_aligned-n_items)/n_targs
            return rew

class ReverseClusterMatchController(EvenLineMatchController):
    """
    This class creates an instance of the inverse of a Cluster Line
    Match game. The agent and targets are reversed.

    The agent must place a cluster of items matching the number of
    target objects. The items must not be all in a single row and
    must not all be aligned with the target columns.
    """
    def calculate_reward(self, harsh: bool=False):
        """
        Determines what reward to return. In this case, checks if
        the same number of items exists as targets and checks that
        all items are not in a single row and that the items do not
        align perfectly in the target columns.
        
        If all of these factors are met, if harsh is true, the
        function returns a reward of 1. If harsh is false, the
        function returns a partial reward based on the difference of
        the number of items to targs divided by the number of targs.
        A 0 is returned if all items are aligned with targs or if all
        items are in a single row.

        Args:
            harsh: bool
                if true, the function returns a reward of 1 when the
                same number of items exists as targs and the items are
                not aligned in a single row with the targ columns.
                If harsh is false, the function returns a partial
                reward based on the difference of the number of items
                to targs divided by the number of targs.
                A 0 is returned in both cases if all items are aligned
                with targs or if all items are in a single row.

                harsh == False:
                    rew = (n_targ - abs(n_items-n_targs))/n_targs
                harsh == True:

                    rew = +1 when n_items == n_targs
                          and
                          n_aligned != n_targs

                    rew = 0 when n_items == n_targs
                          and
                          n_aligned == n_targs

                    rew = -1 otherwise
        Returns:
            rew: float
                the calculated reward
        """
        targs = self.register.targs
        items = self.register.items
        n_targs = len(targs)
        n_items = len(items)
        n_aligned = len(get_aligned_items(
            items=items,
            targs=targs,
            min_row=0
        ))
        if n_aligned == n_targs:
            return int(n_aligned == 1)
        if harsh:
            if n_targs != n_items: return -1
            else: return 1 # n_targs==n_items and n_aligned != n_targs
        return (n_targs - np.abs(n_items-n_targs))/n_targs

class ClusterClusterMatchController(ClusterMatchController):
    """
    Creates a game in which the user attempts to place the same
    number of items on the grid as the number of target objects.
    The target objects are randomly placed and no structure is imposed
    on the placement of the user's items.
    """
    def calculate_reward(self, harsh: bool=False):
        """
        Determines what reward to return. In this case, checks if
        the same number of items exists as targets.
        
        Args:
            harsh: bool
                if true, the function returns a reward of 1 when the
                same number of items exists as targs. -1 otherwise.

                If harsh is false, the function returns a partial
                reward based on the difference of the number of items
                to targs divided by the number of targs.

                harsh == False:
                    rew = (n_targ - abs(n_items-n_targs))/n_targs
                harsh == True:
                    rew = +1 when n_items == n_targs
                    rew = -1 otherwise
        Returns:
            rew: float
                the calculated reward
        """
        targs = self.register.targs
        items = self.register.items
        n_targs = len(targs)
        n_items = len(items)
        if harsh:
            return -1 + 2*int(n_targs == n_items)
        return (n_targs - np.abs(n_items-n_targs))/n_targs

class UnevenLineMatchController(EvenLineMatchController):
    """
    This class creates an instance of an Uneven Line Match game.

    The agent must align a single item along the column of each of the
    target objects. The target objects are unevenly spaced.
    """
    def reset(self, n_targs=None, *args, **kwargs):
        """
        This function should be called everytime the environment starts
        a new episode.

        Args:
            n_targs: int or None
                if int is argued, this will dictate the number of
                target items for the episode
            held_out: bool
                if true, will sample an episode that was held out from
                the non-held out episodes
        """
        self.init_variables(n_targs)
        # randomizes object placement on grid
        self.register.uneven_line_match(
            rand_pdb=self.rand_pdb,
            player_on_pile=self.player_on_pile,
            spacing_limit=self.spacing_limit,
            sym_distr=self.sym_distr
        )
        return self.grid.grid

class OrthogonalLineMatchController(ClusterMatchController):
    """
    This class creates an instance of an Orthogonal Line Match game.

    The agent must align the same number of items as targs. The items
    must be aligned vertically and evenly spaced by 0 if the targs are
    spaced by 0 or items must be spaced by 1 otherwise.
    """
    def reset(self, n_targs=None, *args, **kwargs):
        """
        This function should be called everytime the environment starts
        a new episode.
        """
        self.init_variables(n_targs)
        # randomizes object placement on grid
        self.register.orthogonal_line_match(
            rand_pdb=self.rand_pdb,
            player_on_pile=self.player_on_pile,
            spacing_limit=self.spacing_limit,
            sym_distr=self.sym_distr
        )
        return self.grid.grid

class BriefPresentationController(ClusterMatchController):
    """
    This class creates an instance of the Cluster Line Match game in
    which the presentation of the number of targets is only displayed
    for n_targs frames total.

    The agent must place the same number of items as the number of
    targets that were originally displayed along a single row. The
    targets are randomly distributed about the grid.
    """
    def step(self, direction: int, grab: int):
        """
        Step takes a movement and a grabbing action. The function
        moves the player and any items in the following way.

        This function determines if the targets should be displayed
        anymore based on the total number of steps taken so far.

        Args:
          direction: int [0, 1, 2, 3, 4]
            Check DIRECTIONS to ensure these values haven't changed
                0: no movement
                1: move up (lower row unit)
                2: move right (higher column unit)
                3: move down (higher row unit)
                4: move left (lower column unit)
          grab: int [0,1]
            grab is an action to enable the agent to carry items around
            the grid. when a player is on top of an item, they can grab
            the item and carry it with them as they move. If the player
            is on top of a pile, a new item is created and carried with
            them to the next square.
        
            0: quit grabbing item
            1: grab item. item will follow player to whichever square
              they move to.
        """
        self.n_steps += 1
        info = {
            "is_harsh": self.harsh,
            "n_targs": self.n_targs,
            "n_items": self.register.n_items,
            "n_aligned": len(get_aligned_items(
                items=self.register.items,
                targs=self.register.targs,
                min_row=0
            )),
            "disp_targs":int(self.register.display_targs),
            "is_animating":int(self.is_animating),
            "is_pop": int(self.is_pop()),
            "skipped": 0,
        }
        if self.n_steps > self.n_targs and self.is_animating:
            self.register.make_signal(center_signal=self.center_signal)
            self.register.hide_targs()
            self.is_animating = False
        if self.n_steps < self.n_targs+1:
            grab = 0
            info["n_items"] = self.n_steps-1
        elif self.n_steps == self.n_targs+1:
            grab = 0
            info["n_items"] = self.n_steps-1

        event = self.register.step(direction, grab)

        done = False
        rew = 0
        if event == BUTTON_PRESS:
            rew = self.calculate_reward(harsh=self.harsh)
            done = True
        elif event == FULL:
            done = True
            rew = -1
        elif event == STEP:
            done = False
            rew = 0
        return self.grid.grid, rew, done, info

class NutsInCanController(EvenLineMatchController):
    """
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
    """
    def reset(self, n_targs=None, held_out=False):
        """
        This function should be called everytime the environment starts
        a new episode.

        Args:
            n_targs: int or None
                if int is argued, this will dictate the number of
                target items for the episode
            held_out: bool
                if true, will sample an episode that was held out from
                the non-held out episodes
        """
        self.init_variables(n_targs)

        # randomize object placement on grid, only display one target
        # for first frame. invis_targs is a set
        self.register.cluster_match(
            rand_pdb=self.rand_pdb,
            player_on_pile=self.player_on_pile,
            spacing_limit=self.spacing_limit,
            sym_distr=self.sym_distr,
            held_out=held_out
        )
        self.invis_targs = self.register.targs
        self.targ = None
        for targ in self.invis_targs:
            targ.color = COLORS[DEFAULT]
        self.flashed_targs = []
        self.register.draw_register()
        return self.grid.grid

    def step(self, direction: int, grab: int):
        """
        Step takes a movement and a grabbing action. The function
        moves the player and any items in the following way.

        This function determines if the targets should be displayed
        anymore based on the total number of steps taken so far.

        Args:
          direction: int [0, 1, 2, 3, 4]
            Check DIRECTIONS to ensure these values haven't changed
                0: no movement
                1: move up (lower row unit)
                2: move right (higher column unit)
                3: move down (higher row unit)
                4: move left (lower column unit)
          grab: int [0,1]
            grab is an action to enable the agent to carry items around
            the grid. when a player is on top of an item, they can grab
            the item and carry it with them as they move. If the player
            is on top of a pile, a new item is created and carried with
            them to the next square.
        
            0: quit grabbing item
            1: grab item. item will follow player to whichever square
              they move to.
        """
        self.n_steps += 1
        info = {
            "is_harsh": self.harsh,
            "n_targs": self.n_targs,
            "n_items": self.register.n_items,
            "n_aligned": len(get_aligned_items(
                items=self.register.items,
                targs=self.register.targs,
                min_row=0
            )),
            "disp_targs":int(self.register.display_targs),
            "is_animating":int(self.is_animating),
            "is_pop": int(self.is_pop()),
            "skipped": 0,
        }
        if self.targ is None:
            if self.rand_timing and np.random.random()>=self.timing_p:
                self.n_steps -= 1
                info["skipped"] = 1
            else:
                self.targ = self.invis_targs.pop()
                self.targ.color = COLORS[TARG]
        elif len(self.invis_targs) > 0:
            self.targ.color = COLORS[DEFAULT]
            if self.rand_timing and np.random.random()>=self.timing_p:
                self.n_steps -= 1
                info["skipped"] = 1
            else:
                self.flashed_targs.append(self.targ)
                self.targ = self.invis_targs.pop()
                self.targ.color = COLORS[TARG]
        elif len(self.invis_targs)==0 and self.is_animating:
            self.end_animation()
        event = self.register.step(direction, grab)
        if self.n_steps <= self.n_targs + 1:
            info["n_items"] = self.n_steps-1
        done = False
        rew = 0
        if event == BUTTON_PRESS:
            rew = self.calculate_reward(harsh=self.harsh)
            done = True
        elif event == FULL:
            done = True
            rew = -1
        elif event == STEP:
            done = False
            rew = 0
        return self.grid.grid, rew, done, info

    def calculate_reward(self, harsh=False):
        """
        Determines the reward for the agent.

        Args:
            harsh: bool
                currently there is no difference for the harsh flag on
                the reward calculation.
        """
        return 2*(self.register.n_items == self.n_targs) - 1

    def end_animation(self):
        """
        This is called to clean up the initial flashing sequence and
        to display an object that indicates the player should begin
        its counting.
        """
        self.register.make_signal(center_signal=self.center_signal)
        for targ in self.flashed_targs:
            targ.color = COLORS[TARG]
        self.register.hide_targs()
        self.is_animating = False

class VisNutsController(EvenLineMatchController):
    """
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
    """
    def reset(self, n_targs=None, held_out=False):
        """
        This function should be called everytime the environment starts
        a new episode.

        Args:
            n_targs: int or None
                if int is argued, this will dictate the number of
                target items for the episode
            held_out: bool
                if true, will sample an episode that was held out from
                the non-held out episodes
        """
        self.init_variables(n_targs)
        # randomize object placement on grid, only display one target
        # for first frame. invis_targs is a set
        self.register.cluster_match(
            {self.register.get_signal_coord()},
            rand_pdb=self.rand_pdb,
            player_on_pile=self.player_on_pile,
            spacing_limit=self.spacing_limit,
            sym_distr=self.sym_distr,
            held_out=held_out
        )
        self.invis_targs = self.register.targs
        self.targ = None
        for targ in self.invis_targs:
            targ.color = COLORS[DEFAULT]
        self.flashed_targs = []
        self.register.draw_register()
        return self.grid.grid

    def step(self, direction: int, grab: int):
        """
        Step takes a movement and a grabbing action. The function
        moves the player and any items in the following way.

        This function determines if the targets should be displayed
        anymore based on the total number of steps taken so far.

        Args:
          direction: int [0, 1, 2, 3, 4]
            Check DIRECTIONS to ensure these values haven't changed
                0: no movement
                1: move up (lower row unit)
                2: move right (higher column unit)
                3: move down (higher row unit)
                4: move left (lower column unit)
          grab: int [0,1]
            grab is an action to enable the agent to carry items around
            the grid. when a player is on top of an item, they can grab
            the item and carry it with them as they move. If the player
            is on top of a pile, a new item is created and carried with
            them to the next square.
        
            0: quit grabbing item
            1: grab item. item will follow player to whichever square
              they move to.
        """
        self.n_steps += 1
        info = {
            "is_harsh": self.harsh,
            "n_targs": self.n_targs,
            "n_items": self.register.n_items,
            "n_aligned": len(get_aligned_items(
                items=self.register.items,
                targs=self.register.targs,
                min_row=0
            )),
            "disp_targs":int(self.register.display_targs),
            "is_animating":int(self.is_animating),
            "is_pop": int(self.is_pop()),
            "skipped": 0,
        }
        if self.targ is None:
            if self.rand_timing and np.random.random()>=self.timing_p:
                self.n_steps -= 1
                info["skipped"] = 1
            else:
                self.targ = self.invis_targs.pop()
                self.targ.color = COLORS[TARG]
        elif len(self.invis_targs) > 0:
            if self.rand_timing and np.random.random()>=self.timing_p:
                self.n_steps -= 1
                info["skipped"] = 1
            else:
                self.flashed_targs.append(self.targ)
                self.targ = self.invis_targs.pop()
                self.targ.color = COLORS[TARG]
        elif len(self.invis_targs)==0 and self.is_animating:
            self.end_animation()
        event = self.register.step(direction, grab)
        if self.n_steps <= self.n_targs + 1:
            info["n_items"] = self.n_steps-1
        done = False
        rew = 0
        if event == BUTTON_PRESS:
            rew = self.calculate_reward(harsh=self.harsh)
            done = True
        elif event == FULL:
            done = True
            rew = -1
        elif event == STEP:
            done = False
            rew = 0
        return self.grid.grid, rew, done, info

    def calculate_reward(self, harsh=False):
        """
        Determines the reward for the agent.

        Args:
            harsh: bool
                currently there is no difference for the harsh flag on
                the reward calculation.
        """
        return 2*(self.register.n_items == self.n_targs) - 1

    def end_animation(self):
        """
        This is called to clean up the initial flashing sequence and
        to display an object that indicates the player should begin
        its counting.
        """
        self.register.make_signal(center_signal=self.center_signal)
        for targ in self.flashed_targs:
            targ.color = COLORS[TARG]
        self.is_animating = False
        #self.register.draw_register()

class StaticVisNutsController(VisNutsController):
    """
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
    """
    def reset(self, n_targs=None, held_out=False):
        """
        This function should be called everytime the environment starts
        a new episode.

        Args:
            n_targs: int or None
                if int is argued, this will dictate the number of
                target items for the episode
            held_out: bool
                if true, will sample an episode that was held out from
                the non-held out episodes
        """
        self.init_variables(n_targs)
        # randomize object placement on grid, only display one target
        # for first frame. invis_targs is a set
        self.register.cluster_match(
            {self.register.get_signal_coord()},
            rand_pdb=self.rand_pdb,
            player_on_pile=self.player_on_pile,
            spacing_limit=self.spacing_limit,
            sym_distr=self.sym_distr,
            held_out=held_out
        )
        self.invis_targs = self.register.targs
        #for targ in self.invis_targs:
        #    targ.color = COLORS[TARG]
        self.targ = None
        self.flashed_targs = []
        self.register.draw_register()
        return self.grid.grid

class InvisNController(NutsInCanController):
    """
    This class creates a game in which the environment does not ever
    display the targets. The env also displays the end animation
    signal on the second frame. So, initial frame from reset, signal
    pixel, game...
    """

    def step(self, direction: int, grab: int):
        """
        Initial reset frame is blank with n_items equal to n_targs
        Subsequent frame has signal, player cannot play still. n_items
        is equal to 0.

        Next frame, player can play.
        Step takes a movement and a grabbing action. The function
        moves the player and any items in the following way.

        This function determines if the targets should be displayed
        anymore based on the total number of steps taken so far.

        Args:
          direction: int [0, 1, 2, 3, 4]
            Check DIRECTIONS to ensure these values haven't changed
                0: no movement
                1: move up (lower row unit)
                2: move right (higher column unit)
                3: move down (higher row unit)
                4: move left (lower column unit)
          grab: int [0,1]
            grab is an action to enable the agent to carry items around
            the grid. when a player is on top of an item, they can grab
            the item and carry it with them as they move. If the player
            is on top of a pile, a new item is created and carried with
            them to the next square.
        
            0: quit grabbing item
            1: grab item. item will follow player to whichever square
              they move to.
        """
        self.n_steps += 1
        info = {
            "is_harsh": self.harsh,
            "n_targs": self.n_targs,
            "n_items": self.register.n_items,
            "n_aligned": len(get_aligned_items(
                items=self.register.items,
                targs=self.register.targs,
                min_row=0
            )),
            "disp_targs":int(self.register.display_targs),
            "is_animating":int(self.is_animating),
            "is_pop": int(self.is_pop()),
            "skipped": 0,
        }

        # Initial reset frame is blank with n_items equal to n_targs
        # Subsequent frame has signal, player cannot play still. n_items
        # is equal to 0.
        # Next frame, player can play.
        if self.is_animating:
            info["n_items"] = self.n_targs
            self.register.make_signal(center_signal=self.center_signal)
            self.register.hide_targs()
            self.is_animating = False

        event = self.register.step(direction, grab)
        done = False
        rew = 0
        if event == BUTTON_PRESS:
            rew = self.calculate_reward(harsh=self.harsh)
            done = True
        elif event == FULL:
            done = True
            rew = -1
        elif event == STEP:
            done = False
            rew = 0
        return self.grid.grid, rew, done, info

class VisNController(StaticVisNutsController):
    """
    This class creates a game in which the environment immediately
    displays all targets. The env also displays the end animation
    signal on the second frame. So, initial frame from reset with
    all target items, signal pixel, game...
    """

    def step(self, direction: int, grab: int):
        """
        Initial reset frame is blank with n_items equal to n_targs
        Subsequent frame has signal, player cannot play still. n_items
        is equal to 0.

        Next frame, player can play.
        Step takes a movement and a grabbing action. The function
        moves the player and any items in the following way.

        This function determines if the targets should be displayed
        anymore based on the total number of steps taken so far.

        Args:
          direction: int [0, 1, 2, 3, 4]
            Check DIRECTIONS to ensure these values haven't changed
                0: no movement
                1: move up (lower row unit)
                2: move right (higher column unit)
                3: move down (higher row unit)
                4: move left (lower column unit)
          grab: int [0,1]
            grab is an action to enable the agent to carry items around
            the grid. when a player is on top of an item, they can grab
            the item and carry it with them as they move. If the player
            is on top of a pile, a new item is created and carried with
            them to the next square.
        
            0: quit grabbing item
            1: grab item. item will follow player to whichever square
              they move to.
        """
        self.n_steps += 1
        info = {
            "is_harsh": self.harsh,
            "n_targs": self.n_targs,
            "n_items": self.register.n_items,
            "n_aligned": len(get_aligned_items(
                items=self.register.items,
                targs=self.register.targs,
                min_row=0
            )),
            "disp_targs":int(self.register.display_targs),
            "is_animating":int(self.is_animating),
            "is_pop": int(self.is_pop()),
            "skipped": 0,
        }

        # Initial reset frame is blank with n_items equal to n_targs
        # Subsequent frame displays all targs and phase signal,
        # player cannot play still. n_items is equal to 0.
        # Next frame, player can play.
        if self.is_animating:
            info["n_items"] = self.n_targs
            if not self.rand_timing or np.random.random()<=self.timing_p:
                self.end_animation()
            else: 
                info["skipped"] = 1

        event = self.register.step(direction, grab)
        done = False
        rew = 0
        if event == BUTTON_PRESS:
            rew = self.calculate_reward(harsh=self.harsh)
            done = True
        elif event == FULL:
            done = True
            rew = -1
        elif event == STEP:
            done = False
            rew = 0
        return self.grid.grid, rew, done, info

