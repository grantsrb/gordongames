import numpy as np

class Discrete():
    def __init__(self, n_actions):
        self.dtype = np.int32
        self.n = n_actions
        self.actions = np.arange(self.n, dtype=self.dtype)
        self.shape = self.actions.shape

    def contains(self, argument):
        for action in self.actions:
            if action == argument:
                return True
        return False

    def sample(self):
        return np.random.choice(self.n)

class Box():
    def __init__(self, shape):
        """
        Creates a class loosely mimcing the openAI gym Box class.
        
        Args:
            shape: int or tuple of ints
                if int, gets converted into a tuple (shape,)
        """
        self.dtype = np.float32
        self.shape = shape
        if isinstance(shape, int): self.shape = (shape,)
        self.min = -np.ones(self.shape, dtype=self.dtype)
        self.max =  np.ones(self.shape, dtype=self.dtype)

    def sample(self):
        return 2*(np.random.random(self.shape)-.5)
