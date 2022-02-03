from gordongames.envs.ggames.grid import Grid
from gordongames.envs.ggames.registry import Register
from gordongames.envs.ggames.constants import *
from gordongames.envs.ggames.utils import get_rows_and_cols, get_aligned_items, get_max_row
import numpy as np

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
        """
        if type(targ_range) == int:
            targ_range = (targ_range, targ_range)
        assert targ_range[0] <= targ_range[1]
        assert targ_range[0] >= 0 and targ_range[1] < grid_size[1]
        self._targ_range = targ_range
        self._grid_size = grid_size
        self._pixel_density = pixel_density

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
    def n_targs(self):
        return self.register.n_targs

    @property
    def max_punishment(self):
        return -self.targ_range[1]

    def calculate_reward(self):
        raise NotImplemented

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
        event = self.register.step(direction, grab)
        info = {
            "is_harsh": self.harsh,
            "n_targs": self.n_targs,
            "n_items": self.register.n_items,
            "n_aligned": len(get_aligned_items(
                items=self.register.items,
                targs=self.register.targs,
                min_row=0
            ))
        }
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
        This member must be overridden
        """
        raise NotImplemented

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
            divide=True
        )
        self.register = Register(self.grid, n_targs=1)
        self.harsh = harsh

    def reset(self, n_targs=None):
        """
        This function should be called everytime the environment starts
        a new episode.
        """
        if n_targs is None:
            low, high = self.targ_range
            n_targs = np.random.randint(low, high+1)
        # wipes items from grid and makes/deletes targs
        self.register.reset(n_targs)
        # randomizes object placement on grid
        self.register.even_line_match()
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
    def reset(self, n_targs=None):
        """
        This function should be called everytime the environment starts
        a new episode.
        """
        if n_targs is None:
            low, high = self.targ_range
            n_targs = np.random.randint(low, high+1)
        # wipes items from grid and makes/deletes targs
        self.register.reset(n_targs)
        # randomizes object placement on grid
        self.register.cluster_match()
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
    def reset(self, n_targs=None):
        """
        This function should be called everytime the environment starts
        a new episode.
        """
        if n_targs is None:
            low, high = self.targ_range
            n_targs = np.random.randint(low, high+1)
        # wipes items from grid and makes/deletes targs
        self.register.reset(n_targs)
        # randomizes object placement on grid
        self.register.uneven_line_match()
        return self.grid.grid

class OrthogonalLineMatchController(ClusterMatchController):
    """
    This class creates an instance of an Orthogonal Line Match game.

    The agent must align the same number of items as targs. The items
    must be aligned vertically and evenly spaced by 0 if the targs are
    spaced by 0 or items must be spaced by 1 otherwise.
    """
    def reset(self, n_targs=None):
        """
        This function should be called everytime the environment starts
        a new episode.
        """
        if n_targs is None:
            low, high = self.targ_range
            n_targs = np.random.randint(low, high+1)
        # wipes items from grid and makes/deletes targs
        self.register.reset(n_targs)
        # randomizes object placement on grid
        self.register.orthogonal_line_match()
        return self.grid.grid

class BriefPresentationController(ClusterMatchController):
    """
    This class creates an instance of the Cluster Line Match game in
    which the presentation of the number of targets is only displayed
    for 5 frames at the beginning.

    The agent must place the same number of items as the number of
    targets that were originally displayed along a single row. The
    targets are randomly distributed about the grid.
    """
    def reset(self, n_targs=None):
        """
        This function should be called everytime the environment starts
        a new episode.
        """
        self.n_steps = 0
        if n_targs is None:
            low, high = self.targ_range
            n_targs = np.random.randint(low, high+1)
        # wipes items from grid and makes/deletes targs
        self.register.reset(n_targs)
        # randomizes object placement on grid
        self.register.cluster_match()
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
        if self.n_steps >= DISPLAY_STEPS and self.register.display_targs:
            self.register.make_signal()
            self.register.hide_targs()
        if self.n_steps < DISPLAY_STEPS:
            grab = 0
        event = self.register.step(direction, grab)

        info = {
            "is_harsh": self.harsh,
            "n_targs": self.n_targs,
            "n_items": self.register.n_items,
            "n_aligned": len(get_aligned_items(
                items=self.register.items,
                targs=self.register.targs,
                min_row=0
            ))
        }
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
    def reset(self, n_targs=None):
        """
        This function should be called everytime the environment starts
        a new episode.
        """
        self.n_steps = 0
        if n_targs is None:
            low, high = self.targ_range
            n_targs = np.random.randint(low, high+1)
        # wipes items from grid and makes/deletes targs
        self.register.reset(n_targs)

        # randomize object placement on grid, only display one target
        # for first frame. invis_targs is a set
        self.register.cluster_match()
        self.invis_targs = self.register.targs
        self.targ = self.invis_targs.pop()
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
        if len(self.invis_targs) > 0:
            self.targ.color = COLORS[DEFAULT]
            self.flashed_targs.append(self.targ)
            self.targ = self.invis_targs.pop()
            self.targ.color = COLORS[TARG]
        elif len(self.invis_targs)==0 and self.register.display_targs:
            self.end_animation()
        event = self.register.step(direction, grab)
        info = {
            "is_harsh": self.harsh,
            "n_targs": self.n_targs,
            "n_items": self.register.n_items,
            "n_aligned": len(get_aligned_items(
                items=self.register.items,
                targs=self.register.targs,
                min_row=0
            ))
        }
        if self.n_steps <= self.n_targs:
            info["n_items"] = self.n_steps
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
        self.register.make_signal()
        for targ in self.flashed_targs:
            targ.color = COLORS[TARG]
        self.register.hide_targs()
