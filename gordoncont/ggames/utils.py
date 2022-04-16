import numpy as np
from collections import defaultdict

def get_rows_and_cols(objs: set):
    """
    Finds and returns sets of the row and column values of the
    argued set of objects.

    Args:
        objs: set of GameObjects
    Returns:
        rows: set of ints
            a set of the row values
        cols: set of ints
            a set of the column values
    """
    rows, cols = get_row_and_col_counts(objs)
    return set(rows.keys()), set(cols.keys())

def get_row_and_col_counts(objs: set):
    """
    Finds and returns dicts of the row and column values and their
    corresponding counts.

    Args:
        objs: set of GameObjects
    Returns:
        rows: dict
            keys: int
                the row values
            vals: int
                the number of objects with that row value
        cols: dict
            keys: int
                the col values
            vals: int
                the number of objects with that col value
    """
    rows = dict()
    cols = dict()
    for obj in objs:
        row,col = obj.coord
        if row not in rows: rows[row] = 1
        else: rows[row] += 1
        if col not in cols: cols[col] = 1
        else: cols[col] += 1
    return rows, cols

def get_coord_counts(objs: set):
    """
    Counts the number of occurences of the coordinates each object
    within the sequence.

    Args:
        objs: list like of GameObjects
    Returns:
        counts: dict
            keys: coord tuples
                the coordinates
            vals: int
                the counts
    """
    counts = defaultdict(lambda: 0)
    for obj in objs:
        counts[obj.coord] += 1
    return counts

def get_max_row(objs: set, min_row: int=None, ret_count: bool=False):
    """
    Finds the row in which the majority of objects reside. Returns the
    earliest row in case of equal counts.

    Args:
        objs: set of GameObjects
        min_row: int or None (inclusive)
            determines the minimum row available for counting. if None
            all rows are available.
        ret_count: bool
            if true, returns the count associated with the max row
    Returns:
        max_row: int
            the index of the row with the largest number of objects.
        count: int
            the count of the items along the max_row
    """
    rows, _ = get_row_and_col_counts(objs)
    if min_row is not None:
        for i in range(min_row):
            if i in rows: del rows[i]
    max_row = max_key(rows)
    if ret_count:
        if max_row in rows: return max_row, rows[max_row]
        else: return max_row, 0
    return max_row

def get_unaligned_items(items: set, targs: set, min_row: int=2):
    """
    Returns all items that are not aligned along the majority
    row and do not have a target in their column. Only one item
    for one target is allowed.

    Args:
        items: set of GameObjects
            the items in the register
        targs: set of GameObjects
            the targets in the register
        min_row: int
            the minimum row that is allowed to be the majority row. If
            items are below this row, they are automatically considered
            loners.
    Returns:
        loners: set of GameObjects
            the items that don't have a corresponding target in
            their column or do not align on the row with the
            maximum number of target aligned items
    """
    coord_counts = get_coord_counts(items)
    max_row = get_max_row(items, min_row=min_row)
    _, targ_cols = get_rows_and_cols(targs)

    loners = set()
    for item in items:
        if item.coord[1] not in targ_cols:
            loners.add(item)
        elif item.coord[0] != max_row:
            loners.add(item)
        elif coord_counts[item.coord] > 1:
            coord_counts[item.coord] -= 1
            loners.add(item)
    return loners

def get_aligned_items(items: set, targs: set, min_row: int=1):
    """
    Returns all items that are aligned along the majority row and have
    a target in their column. Only one item for one target is allowed.

    Args:
        items: set of GameObjects
            the items in the register
        targs: set of GameObjects
            the targets in the register
        min_row: int
            the minimum row that is allowed to be the majority row. If
            items are below this row, they are automatically considered
            loners.
    Returns:
        aligneds: set of GameObjects
            the items that have a corresponding target in their column
            while sitting on the row with the maximum number of
            aligned items
    """
    loners = get_unaligned_items(
        items=items,
        targs=targs,
        min_row=min_row
    )
    return items-loners

def max_key(d):
    """
    Finds the maximum value of all the keys in the dict and returns
    the key.

    Args:
        d: dict
    Returns:
        k: key or None
            if no rows have any items, returns None
    """
    max_count = 0
    m_key = None
    for k,v in d.items():
        if v > max_count: m_key = k
    return m_key

def nearest_obj(ref_obj, objs):
    """
    Searches through the objs set for the nearest object to the
    ref_obj.

    Args:
        ref_obj: GameObject
        objs: set or list of GameObjects
    Returns:
        nearest: GameObject
            the closest object to the ref_obj
    """
    ref_coord = np.asarray(ref_obj.coord)
    min_dist = np.inf
    nearest = None
    for obj in objs:
        dist = np.linalg.norm(ref_coord - np.asarray(obj.coord))
        if dist < min_dist:
            min_dist = dist
            nearest = obj
    return nearest

def euc_distance(coord1, coord2):
    """
    Finds the euclidian distance between two coordinates.

    Args:
        coord1: sequence of ints (row, col)
        coord2: sequence of ints (row, col)
    Returns:
        dist: float
    """
    return np.linalg.norm(np.asarray(coord1) - np.asarray(coord2))

def find_empty_space_along_row(register, seed_coord):
    """
    Finds the nearest playable empty space along the row of the seed
    coordinate.

    Args:
        register: Register
        seed_coord: tuple of ints (row, col) in grid units
    Returns:
        coord:
            the nearest empty space along the seed row
    """
    grid = register.grid
    row, seed_col = seed_coord
    try_col = seed_col
    max_cols = 2*register.grid.shape[1]
    count = -1
    while not register.is_empty((row,try_col)) and count < max_cols:
        count+=1
        half = count//2
        if count % 2 == 0: try_col = seed_col + half
        else: try_col = seed_col - half
    coord = (row,try_col)
    if not (grid.col_inbounds(try_col) and register.is_empty(coord)):
        return None
    return coord

def add_coords(coord0, coord1):
    """
    Performs a simple vector addition
    """
    return (coord0[0]+coord1[0], coord0[1]+coord1[1])

