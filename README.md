# Gordon Games

## Description
gordongames is a gym environment for recreating computational versions of games proposed in Peter Gordon's paper [_Numerical Cognition Without Words: Evidence from Amazonia_](https://www.science.org/doi/10.1126/science.1094492). 

## Dependencies
- python3
- pip
- gym
- numpy
- matplotlib

## Installation
1. Clone this repository
2. Navigate to the cloned repository
3. Run command `$ pip install -e ./`
4. add `import gordongames` to the top of your python script
5. make one of the envs with the folowing: `env = gym.make("gordongames-<version here>")`

## Rendering
A common error about matplotlib using `agg` can be fixed by including the following lines in your scripts before calling `.render()`:

    import matplotlib
    matplotlib.use('TkAgg')

If you are experiencing trouble using the `render()` function while using jupyter notebook, insert:

    %matplotlib notebook

before calling `render()`.

## Using gordongames
After installation, you can use gordongames by making one of the gym environments. See the paper [_Numerical Cognition Without Words: Evidence from Amazonia_](https://www.science.org/doi/10.1126/science.1094492) for more details about each game.

#### Environment v0 Even Line Match
Use `gym.make('gordongames-v0')` to create the Line Match game. The agent must match the number of target objects by aligning them within the target columns. Targets are evenly spaced. These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)

#### Environment v1 Cluster Match
Use `gym.make('gordongames-v1')` to create the Cluster Line Match game. The agent must match the number target objects, but the target objects are randomly distributed and the agent must align the items in a row. These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)

#### Environment v2 Orthogonal Line Match
Use `gym.make('gordongames-v2')` to create the Orthogonal Line Match game. The agent must match the number of target objects, but the target objects are aligned vertically whereas the agent must align the items along a single row. These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)

#### Environment v3 Uneven Line Match
Use `gym.make('gordongames-v3')` to create the Uneven Line Match game. The agent must match the target objects by aligning them along each respective target column. The targets are unevenly spaced. These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)

#### Environment v4 Nuts-In-Can
Use `gym.make('gordongames-v4')` to create the Nuts-In-Can game. The agent initially watches a number of target objects get breifly flashed, one-by-one. These targets are randomly distributed about the target area. After the initial flash, each target is no longer visible. After all targets are flashed, the agent must then grab the pile the same number of times as there are targets. These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)

#### Environment v5 Reverse Cluster Match
Use `gym.make('gordongames-v5')` to create the Reverse Cluster Line Match game. The agent must match the number of target objects without aligning them. These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)

#### Environment v6 Cluster Cluster Match
Use `gym.make('gordongames-v6')` to create the Cluster Cluster Match game. The target objects are distributed randomly. The agent must simply match the number of target objects with no structure imposed. These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)

#### Environment v7 Brief Display
Use `gym.make('gordongames-v7')` to create the Brief Display game. This is the same as the Cluster Match variant except that the targets are only displayed for the first few frames of the game. The agent must then match the number of randomly distributed target objects from memory. 

These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)

#### Environment v8 Visible Nuts-In-Can
Use `gym.make('gordongames-v8')` to create the Visible Nuts-in-Can game. This is the same as the Nuts-In-Can variant except that the targets are displayed for the entire episode. 

These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)

#### Environment v9 Navigation Task
Use `gym.make('gordongames-v9')` to create the Navigation Task game. The goal of this game is to return all items to the dispenser and then end the episode. This is a non-numeric game which can be used as a control for visual experience without numeric experience.

These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)

#### Environment v10 Static Visible Nuts-In-Can
Use `gym.make('gordongames-v10')` to create the Static Visible Nuts-in-Can game. This is the same as the Visible Nuts-In-Can variant except that all targets are displayed on the first frame instead of being revealed one-by-one.

These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)

## Game Details
Each game consists of a randomly intitialized grid with various objects distributed on the grid depending on the game type. The goal is for the agent to first complete some task and then press the end button located in the upper right corner of the grid. Episodes last until the agent presses the end button. The agent can move left, up, right, down, or stay still. The agent also has the ability to interact with objects via the grab action. Grab only acts on objects in the same square as the agent. If the object is an "item", the agent carries the item to wherever it moves on that step. If the object is a "pile", a new item is created and carried with the agent for that step. The ending button is pressed using the grab action. The reward is only granted at the end of each episode if the task was completed successfully.

#### Rewards
A +1 reward is returned only in the event of a successful completion of the task.

A -1 reward is returned when a task ends unsuccessfully.

##### Environment v0
The agent receives a +1 reward if each target has a single item located in its column at the end of the episode.

##### Environment v1
The agent receives a +1 reward if there exists a single item for each target. The agent must align the items along a single row.

##### Environment v2
The agent receives a +1 reward if there exists an item for each target. All items must be aligned along a single row.

##### Environment v3
The agent receives a +1 reward if each target has a single item located in its column at the end of the episode.

##### Environment v4
The agent receives a +1 reward if the agent removes the exact number of items placed in the pile.

##### Environment v5
The agent receives a +1 reward if there exists an item for each target. All items must not be aligned with the target objects.

##### Environment v6
The agent receives a +1 reward if there exists an item for each target.

##### Environment v7
The agent receives a +1 reward if there exists a single item for each target. The agent must align the items along a single row.

##### Environment v8
The agent receives a +1 reward if the agent removes the exact number of items placed in the pile.

##### Environment v9
The agent receives a +1 reward if the agent deletes all items placed on the grid. Otherwise -1.

##### Environment v10
The agent receives a +1 reward if the agent removes the exact number of items placed in the pile.


#### Game Options

- `grid_size`: tuple of ints - A row,col coordinate denoting the number of units on the grid (height, width).
- `pixel_density`: int - Number of numpy pixels within a single grid unit.
- `targ_range`: tuple of ints - A range of possible initial target object counts for each game (inclusive). Must be less than `grid_size`. 
- `hold_outs`: set of ints  - a set or list of target counts that should not be considered when creating a new game
- `rand_pdb`: bool - if true, the player, dispenser (aka pile), and ending button are randomly distributed along the top row at the beginning of the game. Otherwise they are deterministically set.
- `sym_distr`: bool - if false and `rand_pdb` is false, the player, dispenser, and button are consistently distributed the same way at initialization on every episode. Otherwise the initial distribution is reflected about the yaxis with 50% prob. Only applies when `rand_pdb` is false.
- `player_on_pile`: bool - if true, the player always starts on top of the dispenser pile in counting games. If false, it will not.
- `spacing_limit`: int - if greater than 0, limits the spacing between the player and dispenser, and the ending button and dispenser to be within `spacing_limit` steps on either side of the dispenser's initial position. If `rand_locs` is false, the player and ending button will always be `spacing_limit` steps away symmetrically centered on the dispenser.
- `rand_timing`: bool - if true, the timing of the initial display phase is stochastic so that the agent cannot simply count the number of frames rather than the number of target items.
- `timing_p`: float between 0 and 1 - the probability of an animation step displaying the next target object. A value of 1 means the agent could count the number of frames instead of the number of target items. A value of 0 will not allow the game to progress past the animation phase.

Each of these options are member variables of the environment and will come into effect after the environment is reset. For example, if you wanted to use 1-5 targets in game A, you can be set this using the following code:

    env = gym.snake('gordongames-v0')
    env.targ_range = (1,5)
    observation = env.reset()
    ...
    # You can specify the number of targets directly at reset
    observation = env.reset( n_targs=5 )


#### Environment Parameter Examples
Examples coming soon!

#### About the Code
Coming soon!
