PLAYER = "player"
TARG = "targ"
PILE = "pile"
ITEM = "item"
DIVIDER = "divider"
SIGNAL = "signal"
BUTTON = "button"
DEFAULT = "default"

OBJECT_TYPES = {
    PLAYER: PLAYER,
    TARG: TARG,
    PILE: PILE,
    ITEM: ITEM,
    DIVIDER: DIVIDER,
    BUTTON: BUTTON,
    SIGNAL: SIGNAL,
}

STAY = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4
GRAB = 5

DIRECTIONS = {
    STAY: STAY,
    UP: UP,
    RIGHT: RIGHT,
    DOWN: DOWN,
    LEFT: LEFT
}

DIRECTION2STR = {
    STAY:  "STAY",
    UP:    "UP",
    RIGHT: "RIGHT",
    DOWN:  "DOWN",
    LEFT:  "LEFT"
}


"""
colors: dict
    a dictionary to indicate what objects have what color
    items: (key: str, val: float)
      targ: the color of the target items
      pile: the color of the pile of items
      item: the color of individual items separated from the pile
      player: the color of the player
      button: the color of the ending button
"""
COLORS = {
    TARG: .4,
    PILE: .20,
    ITEM: .09,
    PLAYER: .69,
    DIVIDER: -.3,
    BUTTON: -.1,
    DEFAULT: 0,
    SIGNAL: -.163
}

"""
The events are used in the game to signal what type of event occurred
at each step.

    STEP: nothing of interest occurred
    BUTTON: the ending button was pressed by the player
    FULL: the grid is full of objects
"""
STEP = 0
BUTTON_PRESS = 1
FULL = 2

EVENTS = {
    STEP: STEP,
    BUTTON_PRESS: BUTTON_PRESS,
    FULL: FULL,
}


PRIORITY2TYPE = {
    1:PILE,
    2:BUTTON,
    3:ITEM,
    4:PLAYER,
    5:TARG,
    6:SIGNAL,
}
TYPE2PRIORITY = {
    PILE:1,
    BUTTON:2,
    ITEM:3,
    PLAYER:4,
    TARG:5,
    SIGNAL:6,
}

