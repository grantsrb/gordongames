from gordongames.envs.ggames.grid import Grid
import math
import time
import numpy as np
from collections import defaultdict
from gordongames.envs.ggames.constants import *

class GameObject:
    """
    The GameObject class is the main class for tracking what types of
    objects are on the grid. It contains the current coordinate and
    type of object.
    """
    def __init__(self,
                 obj_type: str,
                 color: float,
                 coord: tuple=(0,0)):
        """
        obj_type: str
          the type of object. see OBJECT_TYPES for a list of available
          types.
        color: float
          the color of the object
        coord: tuple (row, col) in grid units
          the initial coordinate of the object
        """
        self._obj_type = obj_type
        self._color = color
        self.coord = coord
        self.prev_coord = (-math.inf, -math.inf) # used to track changes for drawing to grid

    def move_to(self, coord):
        """
        Moves the object to the argued coordinate.
        """
        self.prev_coord = self.coord
        self.coord = coord

    @property
    def type(self):
        return self._obj_type

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, new_color):
        self._color = new_color

    def __str__(self):
        return self.type

class Register:
    """
    The register tracks the coordinates of all objects within the game.
    The possible objects are:
      targs - the target items for the game
      items - the moveable items for the game
      piles - the location for players to make and delete items
      players - the agents within the game
      dividers - a dividing line across the middle of the map to
        distinguish playable area from target area
      buttons - a button to press once the player thinks the task is
        finished
    
    The register has two data structures to track the items in the game.
    It has a set called obj_register that holds all of the GameObjects
    that are not dividers. It also has a dict called coord_register
    that maps coordinates to sets of items. Only coordintates with
    items are included in the coord_register.

    The register also performs basic game logic. It handles moving the
    player, prevents illegal moves, and handles item placement and
    item creation (when player grabs from piles).

    If the player was carrying an object and stepped onto another
    object, the game is handled as follows. While the player
    continues to grab, all objects and the player remain overlayn.
    If the player releases the grab button while an object is on
    top of another object, one of 2 things can happen. If the 
    underlying object is a pile, the item is returned to the pile.
    If the underlying object is an item, the previously carried
    item is placed in the nearest empty location from its current
    coord.

    The order of the search is the pixels connected to the argued
    coordinate starting in the upper left connected pixel moving
    across the top to the right, then lower left pixel moving
    across the bottom to the right, then leftmost pixels in
    between the top and bottom moving from top to bottom, then the
    rightmost pixels in between the top and bottom. If none of
    these spaces are free, the search repeats one more layer
    outward.
    """
    def __init__(self,
                 grid: Grid,
                 n_targs: int,
                 n_held_outs=0):
        """
        Creates a player, a pile, and the specified number of targs.
        
        Args:
          grid: Grid
            the grid for the game
          n_targs: int
            the number of targets on the screen
          n_held_outs: int
            the number of held out spaces for each target quantity
        """
        self.grid = grid
        self.player = GameObject(obj_type=PLAYER, color=COLORS[PLAYER])
        self.pile = GameObject(obj_type=PILE, color=COLORS[PILE])
        self.button = GameObject(obj_type=BUTTON, color=COLORS[BUTTON])
        self._targs = self.make_targs(n_targs)
        self.obj_register = {
            self.player,
            self.pile,
            self.button,
            *self._targs
        }
        self.coord_register = dict()
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                coord = (row,col)
                self.coord_register[coord] = set()
        self.coord_register[(0,0)] = set(self.obj_register)
        self.button_event_registry = set()
        self.full_grid_event_registry = set()
        self.display_targs = True
        self.invsbl_list = []
        self.rand = np.random.default_rng(int(time.time()))
        self.n_held_outs = n_held_outs
        self.held_outs = self.get_held_outs(n_held_outs)

    @property
    def n_targs(self):
        return len(self._targs)

    @property
    def n_items(self):
        return len(self.items)

    @property
    def items(self):
        """
        Filters the obj_register for the item type gameobjects

        Returns:
            items: set of GameObjects
        """
        items = set()
        for obj in self.obj_register:
            if obj.type == ITEM: items.add(obj)
        return items

    @property
    def targs(self):
        """
        Filters the obj_register for the targ type gameobjects

        Returns:
            targs: set of GameObjects
        """
        return {*self._targs}

    def get_held_outs(self, n_held_outs=4, center_signal=True):
        """
        Creates a dict of heldouts dependent on the parameters of the
        register.

        Args:
            n_held_outs: int
                the number of held out coordinates for each target
                quantity
        Returns:
            held_outs: dict
                keys: int
                    the target quantities
                vals: set of coords
                    the held out coordinates
        """
        rand = np.random.default_rng(12345)
        if self.grid.is_divided: low = self.grid.middle_row+1
        else: low = 0
        high = self.grid.shape[0]
        max_targ = self.grid.shape[1]
        held_outs = { t: set() for t in range(1, max_targ) }
        if n_held_outs is not None:
            for n in range(n_held_outs):
                coords = {
                    (-1,-1), self.get_signal_coord(center_signal)
                }
                for t in range(1, max_targ):
                    coord = (-1,-1)
                    while coord in coords or coord in held_outs[t] or\
                                 (t-1>0 and coord in held_outs[t-1]) or\
                                 (t-2>0 and coord in held_outs[t-2]):
                        row = rand.integers(low, high)
                        col = rand.integers(0, self.grid.shape[1])
                        coord = (row, col)
                    coords.add(coord)
                    held_outs[t].add(coord)
        return held_outs

    def reset(self, n_targs: None or int=None):
        """
        Resets the grid and draws the register to the grid

        Args:
            n_targs: None or int
                if int, changes the number of targets to match the
                argued value. targs are deleted randomly.
        """
        self.display_targs = True
        self.delete_items(incl_targs=True, incl_signals=True)
        if n_targs is not None: self.initialize_targs(n_targs)
        self.grid.reset() # makes a fresh grid
        self.draw_register()

    def register_button_event_handler(self, fxn):
        """
        Registers a function to be called on a button press event.

        Args:
            fxn: callable function
                the function to be called when a button press event
                occurs
        """
        self.button_event_registry.add(fxn)

    def register_full_grid_event_handler(self, fxn):
        """
        Registers a function to be called on a full grid event.

        Args:
            fxn: callable function
                the function to be called when a full grid event
                occurs
        """
        self.full_grid_event_registry.add(fxn)

    def raise_full_grid_event(self):
        """
        Called when the grid is completely full. Calls all functions
        registered in the full_grid_event_registry
        """
        for fxn in self.full_grid_event_registry:
            fxn()

    def raise_button_event(self):
        """
        Called when the button is pressed. Calls all functions in the
        button_event_registry
        """
        for fxn in self.button_event_registry:
            fxn()

    def make_targs(self, n_targs: int):
        """
        Creates the intial target objects. DOES NOT REGISTER THEM!!
        
        Args:
          n_targs: int
            the number of targets to create
        """
        targs = set()
        for i in range(n_targs):
            targ = GameObject(
              obj_type=TARG,
              color=COLORS[TARG],
              coord=(0,0)
            )
            targs.add(targ)
        return targs

    def initialize_targs(self, n_targs: int):
        """
        Creates or deletes targets from the self._targs set to match
        the argued number of target objects.
        
        Args:
          n_targs: int
            the desired number of targets
        """
        if not hasattr(self, "_targs"):
            self._targs = self.make_targs(n_targs)
        elif len(self._targs) < n_targs:
            n = n_targs - len(self._targs)
            self._targs = {*self._targs, *self.make_targs(n)}
        elif len(self._targs) > n_targs:
            targs = self.targs
            loop_len = len(self._targs)-n_targs
            for i in range(loop_len):
                self.delete_obj(targs.pop())
        self.register_targs()
        return self._targs

    def register_targs(self):
        """
        Used as a failsafe to ensure all targs are registered in both
        the obj and coord registers
        """
        for targ in self._targs:
            self.obj_register.add(targ)
            self.coord_register[targ.coord].add(targ)

    def step(self, direction: int, grab: int):
        """
        Step takes two actions and moves the player and any items
        appropriately.

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
        # If did_move is false, the move was either STAY or illegal.
        # Either way, the player object's prev_coord will be updated
        # to reflect what its coord variable was before entering the
        # move object function
        did_move = self.move_player(direction)

        if grab == 0:
            # must check for overlapping objects and handle
            # appropriately
            event = self.handle_drop(self.player)
        else:
            # if player was grabbing, create new item if previous
            # location was a pile, raise button press event if previous
            # location was a button, and carry item to current
            # coordinate if previous location was an item
            event = self.handle_grab(self.player)
        # Draws all registered objects to the grid and updates their
        # previous coord to their current coord
        self.draw_register()
        return event

    def handle_drop(self, player):
        """
        This function is applied to the argued player's previous
        location. If any objects are overlapping they are handled as
        follows.
            item or targ on pile: obj is deleted
            item or targ on item or targ or button: item/targ is placed
                in nearest empty location. see find_space() for details
                into the order of the search.

        Nothing happens if one or fewer objects resides at the player's
        prev_coord

        Args:
            player: GameObject
        """
        prev_objs = set(self.coord_register[player.prev_coord])
        assert len(prev_objs) < 4
        if len(prev_objs) > 1:
            # track the counts of each object type
            objs = {k: [] for k in OBJECT_TYPES.keys()}
            for obj in prev_objs:
                objs[obj.type].append(obj)
            # If there is an item on the coordinate and there is a
            # pile too, then we delete the item
            if len(objs[ITEM]) > 0 and len(objs[PILE]) > 0:
                self.delete_obj(objs[ITEM][0])
            # If there is a targ on the coordinate and there is a
            # pile too, then we delete the targ
            elif len(objs[TARG]) > 0 and len(objs[PILE]) > 0:
                self.delete_obj(objs[TARG][0])
            # If there is an item or a targ and another item,targ or
            # button, then we find the nearest empty coordinate for
            # one of the items
            elif len(objs[ITEM]) > 1 or\
                    (len(objs[ITEM]) > 0 and\
                        (len(objs[BUTTON]) > 0 or len(objs[TARG]) > 0)):
                free_coord = self.find_space(player.prev_coord)
                if free_coord is not None:
                    self.move_object(objs[ITEM][0], free_coord)
                else:
                    return FULL
            elif len(objs[TARG]) > 1 or\
                    (len(objs[TARG]) > 0 and\
                        (len(objs[BUTTON]) > 0 or len(objs[ITEM]) > 0)):
                free_coord = self.find_space(player.prev_coord)
                if free_coord is not None:
                    self.move_object(objs[TARG][0], free_coord)
                else:
                    return FULL
        return STEP

    def find_space(self, coord, playable_half=True):
        """
        Searches around the argued coord for a coordinate that is
        either empty or only contains a player game object.

        The order of the search is the pixels connected to the argued
        coordinate starting in the upper left connected pixel moving
        across the top to the right, then lower left pixel moving
        across the bottom to the right, then leftmost pixels in
        between the top and bottom moving from top to bottom, then the
        rightmost pixels in between the top and bottom. If none of
        these spaces are free, the search repeats one more layer
        outward.

        Args:
            coord: tuple in grid units (row, col)
                this is the root of the breadth first search. It is
                not included in the search.
            playable_half: bool
                if true, the algorithm only considers locations that
                are in the playable parts the grid. If false, only
                considers spaces that are in unplayable parts of the
                grid.
        Returns:
            free_coord: tuple
                the nearest coordinate that is empty or only contains
                a player object.
        """
        def test_loc(loc):
            """
            loc: tuple coordinate in grid units (row, col)
            """
            if playable_half:
                return self.is_playable(loc) and self.is_empty(loc)
            return self.is_targ_space(loc) and self.is_empty(loc)
        row,col = coord
        layer = 0
        while layer < max(*self.grid.shape):
            layer += 1
            min_row = row - layer
            min_col = col - layer
            max_row = row + layer
            max_col = col + layer
            for i in range(max_col-min_col+1):
                loc = (min_row, min_col+i)
                if test_loc(loc): return loc
                loc = (max_row, min_col+i)
                if test_loc(loc): return loc
            for i in range(1,max_row-min_row):
                loc = (min_row+i, min_col)
                if test_loc(loc): return loc
                loc = (min_row+i, max_col)
                if test_loc(loc): return loc
        self.raise_full_grid_event()
        return None

    def is_empty(self, coord):
        """
        A SPACE IS CONSIDERED EMPTY EVEN IF THE PLAYER OCCUPIES IT!!

        Checks if the argued coordinate is empty of GameObjects
        except for the player object. THE PLAYER OBJECT IS IGNORED!!!

        Args:
            coord: tuple in grid units (row, col)
        """
        coord = tuple(coord)
        if not self.grid.is_inbounds(coord): return False
        objs = self.coord_register[coord]
        if len(objs) == 0: return True
        elif len(objs) == 1 and self.player in objs: return True
        return False

    def is_playable(self, coord):
        """
        Determines if the coord is within playable bounds of the grid
        """
        return self.grid.is_playable(coord)

    def is_targ_space(self, coord):
        """
        Determines if the coord is within the space alotted for targets.
        If the grid is not divided, this means anywhere is fair game.
        If the grid is divided, then only the spaces below the middle
        line return true.
        """
        return self.grid.is_below_divider(coord)

    def is_overlapped(self, coord):
        """
        Checks if the argued coordinate has two or more GameObjects
        other than the player object. THE PLAYER OBJECT IS IGNORED!!!

        A SPACE IS NOT CONSIDERED TO OVERLAP IF THE PLAYER OCCUPIES IT
        WITH ANOTHER OBJECT!! THE PLAYER DOES NOT COUNT IN THIS
        FUNCTION!!

        Args:
            coord: tuple in grid units (row, col)
        Returns:
            is_overlapping: bool
                if true, multiple GameObjects other than the player
                object reside in this space
        """
        coord = tuple(coord)
        if not self.is_playable(coord): return False
        objs = self.coord_register[coord]
        if len(objs) > 2: return True
        elif self.player not in objs and len(objs) > 1: return True
        return False

    def delete_obj(self, game_object: GameObject):
        """
        Deletes the object from the registries and the grid

        Args:
            game_object: GameObject
                the gameobject to be deleted
        """
        self.grid.draw(
            game_object.prev_coord,
            -game_object.color,
            add_color=True
        )
        if game_object.coord in self.coord_register:
            self.coord_register[game_object.coord].remove(game_object)
        self.obj_register.remove(game_object)
        if game_object.type == TARG: self._targs.remove(game_object)
        elif game_object == self.player: del self.player
        elif game_object == self.button: del self.button
        elif game_object == self.pile: del self.pile
        else: del game_object

    def delete_items(self, incl_targs=False, incl_signals=True):
        """
        Deletes all items from the registers.

        Args:
            incl_targs: bool
                if true, targets are also deleted.
            incl_signals: bool
                if true, signals are also deleted.
        """
        reg = {*self.obj_register}
        for obj in reg:
            d = obj.type == ITEM or (incl_targs and obj.type == TARG)
            if d or (incl_signals and obj.type == SIGNAL):
                self.delete_obj(obj)

    def handle_grab(self, player):
        """
        Assumes that a grab action was performed.

        Uses the previous and current coord of the argued player to
        handle any object interactions appropriately.

        The function operates as follows:
            create new item if previous location was a pile,
            raise button press event if previous location was a button,
            carry targ or item to current coordinate if previous
                location was a targ or item

        Args:
            player: GameObject
        """
        # copy set of objects residing in the previous location
        prev_objs = set(self.coord_register[tuple(player.prev_coord)])
        if len(prev_objs) > 0:
            for obj in prev_objs:
                if obj.type == ITEM or obj.type == TARG:
                    self.move_object(obj, coord=player.coord)
                    return STEP
            # Only possibility for 2 objects is if player is one of them
            if len(prev_objs) == 2 and player in prev_objs:
                prev_objs.remove(player)
            obj = prev_objs.pop()
            if obj.type == PILE:
                self.make_object(obj_type=ITEM, coord=player.coord)
            elif obj.type == BUTTON:
                self.raise_button_event()
                return BUTTON_PRESS
        return STEP

    def get_signal_coord(self, center_signal=True):
        """
        Returns the rightmost square one space down from the topmost
        row if center_signal is false. Otherwise returns the center
        square of the lower half of the grid unless
        it is occupied. If occupied this function searches one unit
        left, up, right, then down for a free space. This repeats if
        no free spaces are found.

        Args:
            center_signal: bool
                determines if the signal pixel will go in the middle
                of the grid or one row down from the top row, in the
                rightmost column.
        Returns:
            coord: tuple of ints
                the coordinate in which the signal object goes.
        """
        grid = self.grid
        if center_signal:
            row = int(3*grid.shape[0]/4)
            col = grid.shape[1]//2
            coord = (row,col)
            if not self.is_empty(coord):
                coord = self.find_space(coord, playable_half=False)
            return coord
        else:
            row = 1
            col = grid.shape[1]-1
            return (row,col)

    def make_signal(self, coord=None, center_signal=True):
        """
        Creates a signal object

        Args:
            coord: tuple of ints (row,col) or None
                optional coordinate for the signal
            center_signal: bool
                if true, signal coord will be centered in demonstration
                area. Otherwise will go one row down from the uppermost
                right square of the playable area
        """
        if coord is None:
            coord = self.get_signal_coord(center_signal)
        self.make_object( obj_type=SIGNAL, coord=coord )
        if not center_signal:
            coord = (coord[0],0)
            self.make_object( obj_type=SIGNAL, coord=coord )

    def make_object(self, obj_type: str, coord: tuple):
        """
        Creates a new instance of the argued object type at the argued
        coordinate. Automatically adds the object to the registers

        Args:
            obj_type: str
                the type of object. See OBJECT_TYPES for options
            coord: tuple in grid units (row, col)
                the intial coordinate of the object
        """
        coord = tuple(coord)
        obj = GameObject(
            obj_type=obj_type,
            color=COLORS[obj_type],
            coord=coord
        )
        self.obj_register.add(obj)
        self.coord_register[coord].add(obj)

    def apply_direction(self, coord: tuple, direction: int):
        """
        Changes a coord to reflect the applied direction

        Args:
            coord: tuple grid units (row, col)
            direction: int
                the movement direction. see the DIRECTIONS constant
        Returns:
            new_coord: tuple
                the updated coordinate
        """
        new_coord = tuple(coord)
        if direction == UP:
            new_coord = (coord[0]-1, coord[1])
        elif direction == RIGHT:
            new_coord = (coord[0], coord[1]+1)
        elif direction == DOWN:
            new_coord = (coord[0]+1, coord[1])
        elif direction == LEFT:
            new_coord = (coord[0], coord[1]-1)
        return new_coord

    def move_object(self, game_object: GameObject, coord: tuple=None):
        """
        Takes an object and updates its coordinate to reflect the
        argued coord. Updates are reflected in coord_register and in
        the argued game_object. Does not affect game_object's
        prev_coord value but does update its coord value.

        If object does not move, this function returns False. This
        includes if the action is STAY and successfully completed.

        Args:
            game_object: GameObject
                the game object that is being moved
            coord: tuple or None
                if this is argued and direction is None, the object
                is moved to this coord
        Returns:
            did_move: bool
                if true, the move was legal and object was moved.
                If false, the move was either illegal or did not change
                the game_object's coord value.
        """
        if coord == tuple(game_object.coord):
            return False
        if self.grid.is_inbounds(coord):
            prev = tuple(game_object.coord)
            game_object.coord = tuple(coord)
            if game_object in self.coord_register[prev]:
                self.coord_register[prev].remove(game_object)
            self.coord_register[coord].add(game_object)
            return True
        return False

    def move_player(self, direction):
        """
        Takes a direction and updates the player's coordinate to
        reflect the applied direction. Updates are reflected in
        coord_register and in the player.

        If player does not move, this function returns False. This
        includes if the action is STAY and successfully completed.

        Args:
            direction: int
                the movement direction. See DIRECTIONS constant.
        Returns:
            did_move: bool
                if true, the move was legal and object was moved.
                If false, the move was either STAY and the game_object
                is updated. Or the move was illegal and the
                game_object does not change
        """
        direction = direction % len(DIRECTIONS)
        coord = self.apply_direction(self.player.coord, direction)
        if self.grid.is_playable(coord):
            return self.move_object(self.player, coord)
        else:
            return False

    def draw_register(self):
        """
        This function updates the grid with the current state of the
        registers.

        Each GameObject's prev_coord is updated to the value of coord.

        The draw process wipes the grid to the default value, then for
        each coordinate all GameObjects at that coordinate sum their
        colors together which is then drawn to the grid at that coord.
        """
        # clears all information on the grid but maintains intial
        # ndarray reference self.grid._grid.
        self.grid.clear(remove_divider=False)

        n_rows, n_cols = self.grid.shape
        for row in range(n_rows):
            for col in range(n_cols):
                coord = (row,col)
                if len(self.coord_register[coord]) > 0:
                    color = 0
                    for obj in self.coord_register[coord]:
                        if self.display_targs or obj.type != TARG:
                            color += obj.color
                        obj.prev_coord = tuple(obj.coord)
                    self.grid.draw(coord=coord, color=color)

    def draw_register_changes(self):
        """
        This function only updates the grid with changes made to the
        registered game objects. It searches for differences in 
        a GameObject's prev_coord and coord and updates the grid to
        reflect these changes.

        Each GameObject's prev_coord is updated to the value of coord.
        """
        for obj in self.obj_register:
            if obj.prev_coord != obj.coord:
                # Delete previous value
                self.grid.draw(
                    obj.prev_coord,
                    -obj.color,
                    add_color=True
                )
                if self.display_targs or obj.type!=TARG:
                    # Add new value
                    self.grid_draw(
                        obj.coord,
                        obj.color,
                        add_color=True
                    )
                obj.prev_coord = obj.coord

    def hide_targs(self):
        """
        This function is used to hide the targets all at once. This
        function sets the `self.display_targs` member to False which
        prevents the targets from getting drawn in the register draw
        functions. To draw the targets again, simply set
        `self.display_targs` to True.
        """
        self.display_targs = False
        self.draw_register()

    def place_player_pile_button(self, rand_locs=True,
                                       player_on_pile=False,
                                       spacing_limit=None,
                                       sym_distr=True):
        """
        Places the pile, button, and player randomly or evenly along
        the top row of the grid.

        Args:
            rand_locs: bool
                if false, consistently places the pile and button
                along the top row of the grid with the player on top
                of the pile.
                Otherwise each the player, pile, and button are
                randomly placed in any order along the top row.
            player_on_pile: bool
                if true, the player always starts on top of the dispenser
                pile in counting games. If false, it may or may not.
            spacing_limit: None or int greater than 0
                if greater than 0, limits the spacing between the
                player, dispenser, and ending button to be within
                spacing_limit steps on either side of the dispenser's
                initial position. If rand_locs is false, the ending
                button will always be spacing_limit steps away
            sym_distr: bool
                if false and rand_pdb is false, the player, dispenser,
                and button are consistently distributed the same way
                at initialization on every episode. Otherwise the initial
                distribution is reflected about the yaxis with 50% prob.
                Only applies when rand_pdb is false.
        """
        if rand_locs:
            if spacing_limit is None or spacing_limit<=0:
                cols = self.rand.permutation(self.grid.shape[1])
            else:
                col = self.rand.integers(
                    low=spacing_limit,
                    high=self.grid.shape[1]-spacing_limit,
                )
                cols = np.arange( col-spacing_limit, col+spacing_limit)
                cols[-spacing_limit:] += 1
                cols = self.rand.choice( cols, size=2, replace=False )
                cols = [cols[0], col, cols[1]]
        else:
            cols = Register.even_spacing(self.grid.shape[1], 3)
            if not(spacing_limit is None or spacing_limit<=0):
                cols[0] = cols[1]-spacing_limit
                cols[2] = cols[1]+spacing_limit
            if sym_distr and np.random.random() >= .5:
                cols[0],cols[2] = cols[2],cols[0]
        if player_on_pile:
            cols[0] = cols[1]
        self.move_object(self.player, (0, int(cols[0])))
        self.move_object(self.pile,   (0, int(cols[1])))
        self.move_object(self.button, (0, int(cols[2])))

    def rand_nav_placement(self):
        """
        Places all objects randomly on the board.
        """
        for _ in self._targs:
            self.make_object(obj_type=ITEM, coord=(0,0))
        objs = [*(self.obj_register-self._targs)]
        s = len(objs)
        cols = self.rand.integers(
            low=0,
            high=self.grid.shape[1],
            size=s
        )
        obj_order = [int(x) for x in self.rand.permutation(s)]
        if self.grid.is_divided: high = self.grid.middle_row
        else: high = self.grid.shape[0]
        coords = set()
        for i in obj_order:
            row = self.rand.integers(0,high)
            coord = (row,cols[i])
            n_loops = 0
            while coord is None or not self.is_empty(coord) or\
                                               coord in coords:
                n_loops += 1
                if n_loops > 50:
                    n_loops = 0
                    cols[i] = self.rand.integers(
                        low=0,
                        high=self.grid.shape[1]
                    )
                row = self.rand.integers(0,high)
                coord = (row,cols[i])
            coords.add(coord)
            self.move_object(objs[i], coord=coord)

    def rand_targ_placement(self, reserved_coords=set(),
                                  held_outs=defaultdict(set),
                                  invert=False,
                                  center_signal=True):
        """
        Places the targets randomly on the grid.

        Args:
            reserved_coords: set of coords or dict
                if you wish any spaces to be avoided when placing the
                target objects, you can specify these coordinates in
                the reserved_coords set
            held_outs: dict
                if you wish to reserve spaces specific to a target
                quantity, you can argue a dict with keys corresponding
                to the target quantities and values of sets of
                reserved coordinates for those particular quantities
            invert: bool
                if true, will use the held_out spaces for each
                target quantity
        """
        loop_thresh = 10
        coords = {
            (-1, -1),
             self.get_signal_coord(center_signal),
             *reserved_coords
        }
        if self.grid.is_divided: low = self.grid.middle_row+1
        else: low = 0
        high = self.grid.shape[0]
        assert self.n_targs < (high-low)*self.grid.shape[1]
        for t,targ in enumerate(self.targs):
            t = t + 1
            coord = (-1,-1)
            n_loops = 0
            if invert and len(held_outs[t])>0:
                availables = list(held_outs[t])
                while coord in coords or not self.is_empty(coord):
                    if n_loops<loop_thresh:
                        idx = self.rand.integers(0,len(availables))
                        coord = availables[idx]
                    else:
                        row = self.rand.integers(low, high)
                        col = self.rand.integers(0, self.grid.shape[1])
                        coord = (row, col)
                    n_loops += 1
            else:
                while coord in coords or not self.is_empty(coord)\
                                    or coord in held_outs[t]:
                    row = self.rand.integers(low, high)
                    col = self.rand.integers(0, self.grid.shape[1])
                    coord = (row, col)
            coords.add(coord)
            self.move_object(targ, coord=coord)

    @staticmethod
    def even_spacing(max_int, n):
        """
        Returns a list of indices that denote n evenly spaced indices
        ranging from 0 to max_int inclusive.

        Args:
            max_int: int
                the total space to work with.
        """
        assert n>0
        return [int(i) for i in np.linspace(0,max_int,n+2)[1:-1]]

    def even_targ_spacing(self):
        """
        Evenly spaces the targets by a random amount and places them
        along a random row (below the divider) beginning at a random
        column.
        """
        row = self.rand.integers(
            self.grid.middle_row+1,
            self.grid.shape[0]
        )
        avail_col_space = self.grid.shape[1] - self.n_targs
        max_spacing = avail_col_space//max(self.n_targs-1, 1)
        space_between = 0
        if max_spacing > 0:
            space_between = self.rand.integers(0,max_spacing)
        start_col = 0
        taken_space = self.n_targs + space_between*(self.n_targs-1)
        space_left = self.grid.shape[1]-taken_space
        start_col = self.rand.integers(0,space_left+1)
        for i,targ in enumerate(self._targs):
            col = start_col + i*(space_between+1)
            coord = (row, col)
            self.move_object(targ, coord=coord)

    def uneven_targ_spacing(self, max_spacing=5):
        """
        Unevenly spaces the targets by a random amount and places them
        along a single random row (below the divider).

        Args:
            max_spacing: int
                the maximum spacing that can occur between two targets.
                (inclusive)
        """
        row = self.rand.integers(
            self.grid.middle_row+1,
            self.grid.shape[0]
        )
        avail_col_space = self.grid.shape[1] - self.n_targs
        spacings = []
        for i in range(self.n_targs-1):
            if avail_col_space == 0:
                spacings.append(0)
            else:
                lim = min(max_spacing+1, avail_col_space)
                spacing = self.rand.integers(0,lim)
                spacings.append(spacing)
                avail_col_space -= spacing
        spacings = self.rand.permutation(spacings)
        start_col = self.rand.integers(0,avail_col_space+1)

        for i,targ in enumerate(self._targs):
            if i > 0:
                start_col = start_col + spacings[i-1] + 1
            coord = (row, start_col)
            self.move_object(targ, coord=coord)

    def vertical_targ_spacing(self):
        """
        Evenly spaces the targets by a random amount and places them
        along a random column (below the divider) beginning at a random
        row.
        """
        col = self.rand.integers(
            0,
            self.grid.shape[1]
        )
        if self.grid.is_divided:
            space = self.grid.shape[0] - self.grid.middle_row - 1
            start_row = self.grid.middle_row + 1
        else:
            space = self.grid.shape[0]
            start_row = 0
        avail_row_space = space - self.n_targs
        max_spacing = avail_row_space//max(self.n_targs-1, 1)
        space_between = 0
        if max_spacing > 0:
            space_between = self.rand.integers(0,max_spacing)

        taken_space = self.n_targs + space_between*(self.n_targs-1)
        space_avail = space-taken_space
        start_row = start_row + self.rand.integers(0,space_avail+1)
        for i,targ in enumerate(self._targs):
            row = start_row + i*(space_between+1)
            coord = (row, col)
            self.move_object(targ, coord=coord)

    def navigation_task(self, *args, **kwargs):
        """
        Initialization func for the navigation game.

        The agent must navigate to the target items and return them
        to the dispenser.
        """
        self.rand_nav_placement()
        self.rand_targ_placement()
        self.draw_register()

    def even_line_match(self, rand_pdb=True, player_on_pile=False,
                                             spacing_limit=None,
                                             sym_distr=True):
        """
        Initialization function for the line match game A.

        The agent must align an item along the column of each
        of the target objects

        Args:
            rand_pdb: bool
                if true, the player, dispenser, and button are randomly
                placed along the topmost row at the beginning of each
                episode. Otherwise, they are placed evenly spaced in the
                order player, dispenser, button from left to right.
            player_on_pile: bool
                if true, the player always starts on top of the dispenser
                pile in counting games. If false, it may or may not.
            spacing_limit: None or int greater than 0
                if greater than 0, limits the spacing between the
                player, dispenser, and ending button to be within
                spacing_limit steps on either side of the dispenser's
                initial position. If rand_locs is false, the ending
                button will always be spacing_limit steps away
            sym_distr: bool
                if false and rand_pdb is false, the player, dispenser,
                and button are consistently distributed the same way
                at initialization on every episode. Otherwise the initial
                distribution is reflected about the yaxis with 50% prob.
                Only applies when rand_pdb is false.
        """
        # each is randomly placed in the top row of the grid
        self.place_player_pile_button(
            rand_pdb,
            player_on_pile,
            spacing_limit,
            sym_distr
        )
        self.even_targ_spacing()
        self.draw_register()

    def cluster_match(self, reserved_coords=set(),
                            rand_pdb=True,
                            player_on_pile=False,
                            spacing_limit=None,
                            sym_distr=True,
                            held_out=False):
        """
        Intialization function for the Cluster Match game B.

        The agent must match the number of target objects that are
        randomly distributed about the grid.

        Args:
            reserved_coords: set of coords
                if you wish any spaces to be avoided when placing the
                target objects, you can specify these coordinates in
                the reserved_coords set
            rand_pdb: bool
                if true, the player, dispenser, and button are randomly
                placed along the topmost row at the beginning of each
                episode. Otherwise, they are placed evenly spaced in the
                order player, dispenser, button from left to right.
            player_on_pile: bool
                if true, the player always starts on top of the dispenser
                pile in counting games. If false, it may or may not.
            spacing_limit: None or int greater than 0
                if greater than 0, limits the spacing between the
                player, dispenser, and ending button to be within
                spacing_limit steps on either side of the dispenser's
                initial position. If rand_locs is false, the ending
                button will always be spacing_limit steps away
            sym_distr: bool
                if false and rand_pdb is false, the player, dispenser,
                and button are consistently distributed the same way
                at initialization on every episode. Otherwise the initial
                distribution is reflected about the yaxis with 50% prob.
                Only applies when rand_pdb is false.
            held_out: bool
                if true, will sample an episode that was held out from
                the non-held out episodes
        """
        self.place_player_pile_button(
            rand_pdb,
            player_on_pile,
            spacing_limit,
            sym_distr
        )
        self.rand_targ_placement(
            reserved_coords=reserved_coords,
            held_outs=self.held_outs,
            invert=held_out
        )
        self.draw_register()

    def orthogonal_line_match(self, rand_pdb=True,
                                    player_on_pile=False,
                                    spacing_limit=None,
                                    sym_distr=True):
        """
        Initialization function for the orthogonal line match game C.

        The agent must evenly space an item for each target along a
        single column.

        Args:
            rand_pdb: bool
                if true, the player, dispenser, and button are randomly
                placed along the topmost row at the beginning of each
                episode. Otherwise, they are placed evenly spaced in the
                order player, dispenser, button from left to right.
            player_on_pile: bool
                if true, the player always starts on top of the dispenser
                pile in counting games. If false, it may or may not.
            spacing_limit: None or int greater than 0
                if greater than 0, limits the spacing between the
                player, dispenser, and ending button to be within
                spacing_limit steps on either side of the dispenser's
                initial position. If rand_locs is false, the ending
                button will always be spacing_limit steps away
            sym_distr: bool
                if false and rand_pdb is false, the player, dispenser,
                and button are consistently distributed the same way
                at initialization on every episode. Otherwise the initial
                distribution is reflected about the yaxis with 50% prob.
                Only applies when rand_pdb is false.
        """
        self.place_player_pile_button(
            rand_pdb,
            player_on_pile,
            spacing_limit,
            sym_distr
        )
        self.vertical_targ_spacing()
        self.draw_register()

    def uneven_line_match(self, rand_pdb=True, player_on_pile=False,
                                             spacing_limit=None,
                                             sym_distr=True):
        """
        Initialization function for the uneven line match game D.

        The agent must align an item along the column of each
        of the target objects

        Args:
            rand_pdb: bool
                if true, the player, dispenser, and button are randomly
                placed along the topmost row at the beginning of each
                episode. Otherwise, they are placed evenly spaced in the
                order player, dispenser, button from left to right.
            player_on_pile: bool
                if true, the player always starts on top of the dispenser
                pile in counting games. If false, it may or may not.
            spacing_limit: None or int greater than 0
                if greater than 0, limits the spacing between the
                player, dispenser, and ending button to be within
                spacing_limit steps on either side of the dispenser's
                initial position. If rand_locs is false, the ending
                button will always be spacing_limit steps away
            sym_distr: bool
                if false and rand_pdb is false, the player, dispenser,
                and button are consistently distributed the same way
                at initialization on every episode. Otherwise the initial
                distribution is reflected about the yaxis with 50% prob.
                Only applies when rand_pdb is false.
        """
        self.place_player_pile_button(
            rand_pdb,
            player_on_pile,
            spacing_limit,
            sym_distr
        )
        self.uneven_targ_spacing()
        self.draw_register()

