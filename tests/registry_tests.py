from gordongames.envs.ggames.grid import Grid
from gordongames.envs.ggames.registry import Register
import matplotlib.pyplot as plt
from gordongames.envs.ggames.constants import *
import numpy as np

if __name__ == "__main__":
    # Test placing objects ontop of eachother no button
    grid = Grid((15,31), 10, divide=True)
    register = Register(grid, n_targs=3)
    register.even_line_match()
    register.reset()
    register.move_object(register.pile, (0,0))
    register.move_object(register.button, (0,1))
    coord = (3,15)
    register.move_object(register.player, coord)
    register.player.prev_coord = coord
    plt.imshow(register.grid.grid)
    plt.show()
    for i in range(25):
        register.make_object(obj_type=ITEM, coord=coord)
        register.step(STAY, 0)
        plt.imshow(register.grid.grid)
        plt.show()

    # Test placing objects ontop of eachother with button
    grid = Grid((15,31), 10, divide=True)
    register = Register(grid, n_targs=3)
    register.even_line_match()
    register.reset()
    register.move_object(register.pile, (0,0))
    coord = (3,15)
    register.move_object(register.player, coord)
    register.player.prev_coord = coord
    register.move_object(register.button, coord)
    plt.imshow(register.grid.grid)
    plt.show()
    for i in range(25):
        register.make_object(obj_type=ITEM, coord=coord)
        register.step(STAY, 0)
        plt.imshow(register.grid.grid)
        plt.show()

    # Test placing objects ontop of eachother near edge of map no button
    grid = Grid((15,31), 10, divide=True)
    register = Register(grid, n_targs=3)
    register.even_line_match()
    register.reset()
    register.move_object(register.pile, (0,0))
    register.move_object(register.button, (0,1))
    coord = (0,15)
    register.move_object(register.player, coord)
    register.player.prev_coord = coord
    plt.imshow(register.grid.grid)
    plt.show()
    for i in range(25):
        register.make_object(obj_type=ITEM, coord=coord)
        register.step(STAY, 0)
        plt.imshow(register.grid.grid)
        plt.show()

    # Test placing objects ontop of eachother near edge of map with button
    grid = Grid((15,31), 10, divide=True)
    register = Register(grid, n_targs=3)
    register.even_line_match()
    register.reset()
    register.move_object(register.pile, (0,0))
    coord = (0,15)
    register.move_object(register.player, coord)
    register.move_object(register.button, coord)
    register.player.prev_coord = coord
    plt.imshow(register.grid.grid)
    plt.show()
    for i in range(25):
        register.make_object(obj_type=ITEM, coord=coord)
        register.step(STAY, 0)
        plt.imshow(register.grid.grid)
        plt.show()

    grid = Grid((15,31), 10, divide=True)
    register = Register(grid, n_targs=3)
    for i in range(10):
        register.even_line_match()
        register.reset()
        for targ in register.targs:
            print(targ.prev_coord, targ.coord)
        for move in [UP, RIGHT, RIGHT, DOWN, DOWN, DOWN, LEFT, LEFT, LEFT, UP]:
            register.step(move, 0)
            plt.imshow(register.grid.grid)
            plt.show()

    grid.reset()
    register = Register(grid, n_targs=7)
    for i in range(10):
        register.even_line_match()
        register.reset()
        plt.imshow(register.grid.grid)
        plt.show()

    grid.reset()
    register = Register(grid, n_targs=25)
    for i in range(10):
        register.even_line_match()
        register.reset()
        plt.imshow(register.grid.grid)
        plt.show()

    print("uneven")
    grid.reset()
    register = Register(grid, n_targs=3)
    for i in range(10):
        register.uneven_line_match()
        register.reset()
        plt.imshow(register.grid.grid)
        plt.show()

    grid.reset()
    register = Register(grid, n_targs=7)
    for i in range(10):
        register.uneven_line_match()
        register.reset()
        plt.imshow(register.grid.grid)
        plt.show()

    grid.reset()
    register = Register(grid, n_targs=25)
    for i in range(10):
        register.uneven_line_match()
        register.reset()
        plt.imshow(register.grid.grid)
        plt.show()
