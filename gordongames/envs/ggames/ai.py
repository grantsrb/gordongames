import numpy as np
from gordongames.envs.ggames.utils import nearest_obj, euc_distance, get_unaligned_items, get_aligned_items, get_rows_and_cols, get_row_and_col_counts, get_max_row, find_empty_space_along_row
from gordongames.envs.ggames.constants import *

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
        direction: int [0, 1, 2, 3, 4]
          The direction to move closer to coord1 from coord0.
          Check DIRECTIONS to ensure these values haven't changed
              0: no movement
              1: move up (lower row unit)
              2: move right (higher column unit)
              3: move down (higher row unit)
              4: move left (lower column unit)
    """
    if rand is None: rand = np.random
    start_row, start_col = coord0
    end_row, end_col = coord1
    row_diff = int(end_row - start_row)
    col_diff = int(end_col - start_col)
    if row_diff != 0 and col_diff != 0:
        if rand.random() < .5:
            return DOWN if row_diff > 0 else UP
        else:
            return RIGHT if col_diff > 0 else LEFT
    elif row_diff != 0:
        return DOWN if row_diff > 0 else UP
    elif col_diff != 0:
        return RIGHT if col_diff > 0 else LEFT
    else:
        return STAY

def navigation_task(contr):
    """
    Finds the items on the grid and brings them back to the dispenser
    to delete them. Then agent ends the game when no items are left.
    """
    register = contr.register
    player = register.player
    items = register.items

    # determine which object we should grab next
    if len(items) == 0:
        grab_obj = register.button
    else:
        grab_obj = nearest_obj(player, items)

    # if on top of grab_obj, grab it, otherwise don't
    if grab_obj.coord == player.coord: grab = True
    else: grab = False

    # determine where to move next
    if not grab: goal_coord = grab_obj.coord
    # If we're on top of the button, simply issue a STAY order
    elif grab_obj == register.button: return STAY, grab
    elif len(items) > 0:
        goal_coord = register.pile.coord
        if player.coord == goal_coord:
            return STAY, False

    direction = get_direction(
        player.coord,
        goal_coord,
        register.rand
    )
    return direction, grab


def even_line_match(contr):
    """
    Takes a register and finds the optimal movement and grab action
    for the state of the register.

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
        return STAY, 0

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
    if not grab: goal_coord = grab_obj.coord
    # If we're on top of the button, simply issue a STAY order
    elif grab_obj == register.button: return STAY, grab
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

    direction = get_direction(
        player.coord,
        goal_coord,
        register.rand
    )
    return direction, grab

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
        return STAY, 0

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

    if not grab:
        goal_coord = grab_obj.coord
    else: # Either on pile or unaligned object
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
    direction = get_direction(
        player.coord,
        goal_coord,
        register.rand
    )
    return direction, grab

def brief_display(contr):
    """
    Same as cluster_match but issues stay order if display is still
    visible and the agent is on top of a pile or button.
    """
    reg = contr.register
    if contr.is_animating and reg.player.coord == reg.pile.coord:
        return STAY, 0
    else:
        return cluster_match(contr)

def nuts_in_can(contr):
    """
    Takes a register and finds the optimal movement and grab action
    for the state of the register in the nuts in a can game.

    Args:
        contr: Controller
    Returns:
        direction: int
            a directional movement
        grab: int
            whether or not to grab
    """
    reg = contr.register
    player = reg.player
    items = reg.items
    n_targs = reg.n_targs

    if contr.is_animating and reg.player.coord == reg.pile.coord:
        return STAY, 0

    if reg.n_items < n_targs:
        direction = get_direction(player.coord,reg.pile.coord,reg.rand)
        grab = player.coord == reg.pile.coord
    else:
        direction=get_direction(player.coord,reg.button.coord,reg.rand)
        grab = player.coord==reg.button.coord
    return direction, grab

def nonnumeric_nuts_in_can(contr):
    """
    Takes a register and finds the optimal movement and grab action
    for the state of the register in the nuts in a can game if the
    player completely ignored the quantity everytime.
    The optimal policy simply outputs STAY during
    the demonstration phase, and then navigates to the end button and
    presses it during the response phase. The actions are the same
    regardless of the initial display.

    Args:
        contr: Controller
    Returns:
        direction: int
            a directional movement
        grab: int
            whether or not to grab
    """
    reg = contr.register
    player = reg.player
    items = reg.items
    n_targs = reg.n_targs

    if contr.is_animating and reg.player.coord == reg.pile.coord:
        if not contr.prev_skipped: # press button when new items appear
            if contr.n_steps>0: # prev skipped defaults to 0 at start
                return STAY, 1
        return STAY, 0

    direction=get_direction(player.coord,reg.button.coord,reg.rand)
    grab = player.coord==reg.button.coord
    return direction, grab

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
        return STAY, 0
    
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
    elif grab_obj == register.button: return STAY, grab
    elif len(items) > len(targs):
        goal_coord = register.pile.coord
    # Here we know that we have an item in our grasp. if we're in an
    # empty space, we can simply drop it. And that will have already
    # been done in the logic above. We can just search for the nearest
    # empty space centered on the pile.
    else:
        goal_coord = register.find_space(register.pile.coord)
    direction = get_direction(player.coord, goal_coord, register.rand)
    return direction, grab

