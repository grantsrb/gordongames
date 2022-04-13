from gordongames.envs.ggames.grid import Grid
import matplotlib.pyplot as plt
from gordongames.envs.ggames.constants import PLAYER, TARG, PILE, ITEM, DIVIDER, BUTTON, OBJECT_TYPES, STAY, UP, RIGHT, DOWN, LEFT, DIRECTIONS, COLORS, EVENTS, STEP, BUTTON, FULL, DEFAULT
import numpy as np

if __name__=="__main__":
    grid = Grid(31, pixel_density=1, divide=False)
    assert np.array_equal(grid.grid, np.zeros((31,31)))
    assert grid.shape == (31,31)
    assert grid.pixel_shape == (31,31)
    assert grid.density == 1
    assert grid.is_divided == False
    print(grid.middle_row)
    assert grid.middle_row == 16
    assert grid.units2pixels((10,15)) == (10,15)
    assert grid.pixels2units((10,15)) == (10,15)
    assert grid.is_inbounds((2,3))
    assert not grid.is_inbounds((0,31))
    assert not grid.is_inbounds((31,0))
    assert grid.is_inbounds((0,30))
    assert grid.is_inbounds((30,0))
    assert grid.is_playable((1,5))
    assert grid.is_playable((20,5))
    assert grid.is_playable((0,30))
    assert grid.is_playable((30,0))
    assert not grid.is_playable((-1,0))
    assert not grid.is_playable((0,-2))
    assert not grid.is_playable((31,5))
    assert not grid.is_playable((5,31))
    grid.draw((1,2), color=1)
    assert grid.grid[1,2] == 1
    grid.draw((1,2), color=2)
    assert grid.grid[1,2] == 3
    grid.draw((1,2), color=2, add_color=False)
    assert grid.grid[1,2] == 2
    grid.clear_unit((1,2))
    assert grid.grid[1,2] == 0
    grid.slice_draw((1,2), (3,5), color=1)
    assert np.array_equal(grid.grid[1:3,2:5], np.ones((2,3)))
    assert not np.array_equal(grid.grid, np.ones_like(grid.grid))
    grid.clear()
    assert np.array_equal(grid.grid, np.zeros_like(grid.grid))

    grid = Grid(31, pixel_density=1, divide=True)
    assert grid.shape == (31,31)
    assert grid.pixel_shape == (31,31)
    assert grid.density == 1
    assert grid.is_divided == True
    assert grid.middle_row == 16
    assert grid.is_inhalfbounds((0,15))
    assert grid.is_inhalfbounds((1,5))
    assert grid.is_playable((1,5))
    assert not grid.is_playable((17,0))
    assert not grid.is_playable((16,0))
    assert grid.is_playable((4,17))

    grid = Grid(30, pixel_density=1, divide=True)
    plt.imshow(grid.grid)
    plt.show()
    arr = np.zeros((30,30))
    arr[16,:] = COLORS[DIVIDER]
    assert np.array_equal(grid.grid, arr)
    assert grid.shape == (30,30)
    assert grid.pixel_shape == (30,30)
    assert grid.density == 1
    assert grid.is_divided == True
    assert grid.middle_row == 16

    grid = Grid(31, pixel_density=10, divide=False)
    assert np.array_equal(grid.grid, np.zeros((310,310)))
    assert grid.shape == (31,31)
    assert grid.pixel_shape == (310,310)
    assert grid.density == 10
    assert grid.is_divided == False
    assert grid.middle_row == 16
    assert grid.units2pixels((10,15)) == (100,150)
    assert grid.pixels2units((10,15)) == (1,1)
    assert grid.pixels2units((105,9)) == (10,0)

    grid = Grid(31, pixel_density=10, divide=True)
    plt.imshow(grid.grid)
    plt.show()
    assert grid.shape == (31,31)
    assert grid.pixel_shape == (310,310)
    assert grid.density == 10
    assert grid.middle_row == 16

    grid.draw((1,2), color=1)
    assert grid.grid[10,20] == 1
    assert grid.grid[19,29] == 0
    grid.draw((11,3), color=.5)
    grid.draw((11,4), color=1)
    plt.imshow(grid.grid)
    plt.show()
    grid.clear_unit((1,2))
    grid.clear_unit((11,3))
    grid.clear_unit((11,4))
    assert not np.array_equal(grid.grid, np.zeros(grid.pixel_shape))
    assert np.array_equal(grid.grid[:159,:], np.zeros((159,310)))

    grid.slice_draw((1,2), (3,5), color=1)
    grid.slice_draw((1,7), (1,9), color=1)
    grid.slice_draw((5,7), (8,7), color=.4)
    grid.slice_draw((-1, 40), (4,5), color=1)
    grid.slice_draw((4,5), (1,2), color=1)
    plt.imshow(grid.grid)
    plt.show()
    grid.clear(remove_divider=False)
    assert not np.array_equal(grid.grid, np.zeros_like(grid.grid))
    grid.slice_draw((1,2), (3,5), color=1)
    grid.slice_draw((20,2), (25,5), color=1)
    grid.clear_playable_space()
    plt.imshow(grid.grid)
    plt.show()
