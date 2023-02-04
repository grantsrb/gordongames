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

### Info dict
The `info` dict returns a number of useful bits of information about
the state of the game. In general, the info dict reflects the state of
the game for the previous visual observation. This is because during
data collection in many RL work flows, the final visual frame is ignored
but the final reward is important. So, often the game data is collecte
so that we keep the first visual obs and ignore the last visual obs but
want to collect all the other data starting from the second visual obs
through the last visual obs. All this is to say, if there was 1 item
on the grid in frame produced by `env.reset()`, the info dict for the
next frame will have a value `info["n_items"] == 1`. Here is a breakdown
of all the information keys.

    `is_harsh`: bool
        a boolean indicating if the reward system is in harsh mode
    `n_targs`: int
        the target quantity for this episode
    `n_items`: int
        The number of items on the grid. There are a few quirks about
        this value. First, during the animation phase, this value
        represents the number of target items that have been displayed
        so far. After the animation phase, this value represents the
        number of play items have been dispensed to the grid by the
        player.
    `n_aligned`: int
        the number of target items that have a corresponding, aligned
        (along the same column) play items.
    `disp_targs`: int
        a binary int representing if the target objects are visually
        displayed or not.
    `is_animating`: int
        a binary int representing if the animation phase is occuring.
        The player cannot GRAB during the animation phase.
    `is_pop`: int
        a binary int representing if the player is currently on top of
        the item dispensing pile. pop stands for player on pile. 
    `min_play_area`: bool
        if true, minimizes the play area (area above the
        dividing line of the grid) to 4 rows. Otherwise,
        dividing line is placed at approximately the middle
        row of the grid.

#### Environment v0 Even Line Match
Use `gym.make('gordongames-v0')` to create the Line Match game. The agent must match the number of target objects by aligning them within the target columns. Targets are evenly spaced. 

The default params:

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)

#### Environment v1 Cluster Match
Use `gym.make('gordongames-v1')` to create the Cluster Line Match game. The agent must match the number target objects, but the target objects are randomly distributed and the agent must align the items in a row. 

#### Environment v2 Orthogonal Line Match
Use `gym.make('gordongames-v2')` to create the Orthogonal Line Match game. The agent must match the number of target objects, but the target objects are aligned vertically whereas the agent must align the items along a single row. 

#### Environment v3 Uneven Line Match
Use `gym.make('gordongames-v3')` to create the Uneven Line Match game. The agent must match the target objects by aligning them along each respective target column. The targets are unevenly spaced. 

#### Environment v4 Nuts-In-Can
Use `gym.make('gordongames-v4')` to create the Nuts-In-Can game. The agent initially watches a number of target objects get breifly flashed, one-by-one. These targets are randomly distributed about the target area. After the initial flash, each target is no longer visible. After all targets are flashed, the agent must then grab the pile the same number of times as there are targets. 

#### Environment v5 Reverse Cluster Match
Use `gym.make('gordongames-v5')` to create the Reverse Cluster Line Match game. The agent must match the number of target objects without aligning them. 

#### Environment v6 Cluster Cluster Match
Use `gym.make('gordongames-v6')` to create the Cluster Cluster Match game. The target objects are distributed randomly. The agent must simply match the number of target objects with no structure imposed. 

#### Environment v7 Brief Display
Use `gym.make('gordongames-v7')` to create the Brief Display game. This is the same as the Cluster Match variant except that the targets are only displayed for the first few frames of the game. The agent must then match the number of randomly distributed target objects from memory. 


#### Environment v8 Visible Nuts-In-Can
Use `gym.make('gordongames-v8')` to create the Visible Nuts-in-Can game. This is the same as the Nuts-In-Can variant except that the targets are displayed for the entire episode. 


#### Environment v9 Navigation Task
Use `gym.make('gordongames-v9')` to create the Navigation Task game. The goal of this game is to return all items to the dispenser and then end the episode. This is a non-numeric game which can be used as a control for visual experience without numeric experience.


#### Environment v10 Static Visible Nuts-In-Can
Use `gym.make('gordongames-v10')` to create the Static Visible Nuts-in-Can game. This is the same as the Visible Nuts-In-Can variant except that all targets are displayed on the first frame instead of being revealed one-by-one.


#### Environment v11 Invis N
Use `gym.make('gordongames-v11')` to create the Give N game. This game never displays the target quantities in the episode. The animation phase ends on the first step of the game. The user of this code must use the `n_targs` value from the info dict to indicate the target quantity to an agent.

#### Environment v12 Vis N
Use `gym.make('gordongames-v12')` to create the Vis N game. This is the same as the Give N variant except that the targets are visible and displayed on the first frame. This is also the same as the Static Visible Nuts-in-Can game except that the animation phase ends on the first frame.


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
- `max_steps`:  positive valued int or None - the maximum number of steps an episode can take.
- `zipf_exponent`: float or None - if not None, the target quantities are sampled
                proportionally to the zipfian distribution with the
                argued exponent. p = 1/(n^z) where n is the target
                quantity, z is the zipfian exponent and p is the
                likelihood.

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
