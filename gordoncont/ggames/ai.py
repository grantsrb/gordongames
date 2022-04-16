import numpy as np
from gordoncont.ggames.utils import nearest_obj, euc_distance, get_unaligned_items, get_aligned_items, get_rows_and_cols, get_row_and_col_counts, get_max_row, find_empty_space_along_row
from gordoncont.ggames.constants import *

def get_even_line_goal_coord(player: object,
                             aligned_items: set,
                             targs: set,
                             max_row: int):
    """
    Finds the row and col of the goal. If no row is established, it
    arbitrarily picks row 2. Assumes len(aligned_items) <= len(targs)

    Args:
        player: GameObject
            the player object
        aligned_items: set of GameObjects
            a set of all the items that are in their final state or
            are at least aligned on the goal row
        targs: set of GameObjects
            a set of all the targs in the game
        max_row: int (exclusive)
            the maximum allowed row if the item row has not yet been
            established. 
    Returns:
        goal_coord: tuple (row, col)
            the coordinate that the player should move towards
    """
    loners = get_unaligned_items(targs, aligned_items)
    goal_targ = nearest_obj(player, loners)
    goal_row = 2
    if len(aligned_items) > 0:
        goal_row = next(iter(aligned_items)).coord[0]
    return (goal_row, goal_targ.coord[1])

def get_direction(coord0, coord1, rand=None):
    """
    Finds the movement direction from coord0 to coord1.

    Args:
        coord0: tuple (row, col) in grid units
            the coordinate that we will be moving from
        coord1: tuple (row, col) in grid units
            the coordinate that we will be moving to
        rand: numpy random number generator
            if none, defaults to numpy.random
    Returns:
        coord: tuple of ints (row,col)
    """
    if rand is None: rand = np.random
    start_row, start_col = coord0
    end_row, end_col = coord1
    row_diff = int(end_row - start_row)
    col_diff = int(end_col - start_col)
    return (row_diff, col_diff)

def even_line_match(contr):
    """
    Takes a register and finds the optimal movement and grab action
    for the state of the register.

    Args:
        contr: Controller
    Returns:
        xycoord: tuple of floats ranging from [-1,1] (lateral,vertical)
            the movement coordinates
        grab: int
            whether or not to grab
    """
    register = contr.register
    player = register.player
    items = register.items
    targs = register.targs
    if contr.is_animating and player.coord == register.pile.coord:
        if register.egocentric:
            return (0,0), 0
        return register.coord2xy(player.coord), 0

    # find items that are out of place
    lost_items = get_unaligned_items(items, targs)
    aligned_items = items-lost_items # set math

    # determine which object we should grab next
    if len(items) == len(targs) and len(lost_items) == 0:
        grab_obj = register.button
    elif len(lost_items) > 0:
        grab_obj = nearest_obj(player, lost_items)
    else:
        grab_obj = register.pile

    # if on top of grab_obj, grab it, otherwise don't
    if grab_obj.coord == player.coord: grab = True
    else: grab = False

    # determine where to move next
    if not grab or grab_obj == register.button:
        goal_coord = grab_obj.coord
    elif len(items) > len(targs):
        goal_coord = register.pile.coord
    # Need to find nearest target that has not been completed and
    # find the appropriate coord for placing an item to complete it
    else:
        try:
            goal_coord = get_even_line_goal_coord(
                player,
                aligned_items,
                targs,
                register.grid.middle_row
            )
        except:
            print("lost items")
            print(lost_items)
            print("aligned items")
            print(aligned_items)
            print("grab_obj")
            print(grab_obj)
            print("items")
            print(items)
            print("targs")
            print(targs)

    if register.egocentric:
        goal_coord = get_direction(
            player.coord,
            goal_coord,
            register.rand
        )
    return register.coord2xy(goal_coord), grab

def cluster_match(contr):
    """
    Takes a register and finds the optimal movement and grab action
    for the state of the register in the cluster match game. Also works
    for cluster cluster match and orthogonal line match.

    Args:
        contr: Controller
    Returns:
        direction: int
            a directional movement
        grab: int
            whether or not to grab
    """
    register = contr.register
    player = register.player
    items = register.items
    n_targs = register.n_targs

    if contr.is_animating and player.coord == register.pile.coord:
        if register.egocentric:
            return (0,0), 0
        return register.coord2xy(player.coord), 0

    min_row = 2
    max_row, n_aligned = get_max_row(
        items,
        min_row=min_row,
        ret_count=True
    )
    unaligned = set(filter(lambda x: x.coord[0]!=max_row, items))
    # check if two objects are ontop of eachother (excluding player)
    is_overlapping = register.is_overlapped(player.coord)
    if is_overlapping:
        grab_obj = nearest_obj(player, items)
    elif n_aligned == n_targs and len(unaligned) == 0:
        grab_obj = register.button
    elif len(unaligned)>0:
        grab_obj = nearest_obj(player, unaligned)
    else:
        grab_obj = register.pile

    if player.coord==grab_obj.coord: grab = True
    else: grab = False

    if not grab or grab_obj == register.button:
        goal_coord = grab_obj.coord
    else: # on pile or unaligned object
        goal_row = max_row if max_row is not None else 2
        temp_col = register.grid.shape[1]//2
        seed_coord = (goal_row, player.coord[1])
        goal_coord = find_empty_space_along_row(
            register=register,
            seed_coord=seed_coord
        )
        # Fail safe in case agent has filled entire row
        # This really shouldn't happen
        if goal_coord is None:
            goal_coord = register.button.coord
            if player.coord == goal_coord: grab = True
            print("Goal coord is None, for seed_coord:", seed_coord)
            print("Item Count:", register.n_items)
            print("Targ Count:", register.n_targs)
    if register.egocentric:
        goal_coord = get_direction(
            player.coord,
            goal_coord,
            register.rand
        )
    return register.coord2xy(goal_coord), grab

def brief_display(contr):
    """
    Same as cluster_match
    """
    return cluster_match(contr)

def nuts_in_can(contr):
    """
    Same as cluster_match
    """
    return cluster_match(contr)

def rev_cluster_match(contr):
    """
    Takes a register and finds the optimal movement and grab action
    for the state of the register in the reverse cluster match game.

    Args:
        contr: Controller
    Returns:
        direction: int
            a directional movement
        grab: int
            whether or not to grab
    """
    register = contr.register
    player = register.player
    items = register.items
    targs = register.targs

    if contr.is_animating and player.coord == register.pile.coord:
        if register.egocentric:
            return (0,0), 0
        return register.coord2xy(player.coord), 0
    
    # used later to determine if all items are aligned
    aligned_items = get_aligned_items(items, targs, min_row=0)

    # check if two objects are ontop of eachother (excluding player)
    is_overlapping = register.is_overlapped(player.coord)
    if len(items) == len(targs) and not is_overlapping:
        if len(targs) == 1 or len(aligned_items)!=len(targs):
            grab_obj = register.button # hard work done
        else: # len(aligned_items) == len(targs)
            # need to unalign an item
            grab_obj = nearest_obj(player, aligned_items)
    elif len(items) >= len(targs) or is_overlapping:
        grab_obj = nearest_obj(player, items) # grab existing item
    else:
        grab_obj = register.pile

    # if on top of grab_obj, grab it, otherwise don't
    if grab_obj.coord == player.coord: grab = True
    else: grab = False

    # determine where to move next
    if not grab: goal_coord = grab_obj.coord
    # If we're on top of the button, simply issue a STAY order and grab
    elif grab_obj == register.button: 
        return register.coord2xy(player.coord), grab
    elif len(items) > len(targs):
        goal_coord = register.pile.coord
    # Here we know that we have an item in our grasp. if we're in an
    # empty space, we can simply drop it. And that will have already
    # been done in the logic above. We can just search for the nearest
    # empty space centered on the pile.
    else:
        goal_coord = register.find_space(register.pile.coord)
    if register.egocentric:
        goal_coord = get_direction(
            player.coord,
            goal_coord,
            register.rand
        )
    return register.coord2xy(goal_coord), grab

