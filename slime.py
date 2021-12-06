# Raphael Laude
# Nov 28, 2021
# slime.py
# License: MIT

import numpy as np, matplotlib, matplotlib.pyplot as plt

# Animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
from matplotlib import rc
rc('animation', html='jshtml')

# Constats #####################################################################

rook = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
queen = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

full = np.ones((3, 3))
el = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
row = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
rotor = np.array([0.5, -0.5])
possible_rook_directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
possible_queen_directions = np.array([[-1, -1], [1, 1], [1, -1], [-1, 1]])

# Helpers ######################################################################

def check_tuple(t, n):
    '''Check that t is a tuple of length n containing only integers.
    Helper for the PetriDish and SlimeAgent classes.
    '''
    assert isinstance(t, tuple), f'{n} must be a tuple.'
    assert len(t) == 2, f'{n} must be a tuple of length 2.'
    for x in t:
        assert isinstance(x, int), f'{n} must contain integers'

def check_np_pair(a):
    '''Check that a is a numpy array of shape (2,) containing only integers.
    Helper for the PetriDish and SlimeAgent classes.
    '''
    assert isinstance(a, np.ndarray), 'array pair must be a numpy array.'
    assert a.shape == (2,), 'array pair must have shape (2,).'
    for x in a:
        assert isinstance(x, np.int64), f'array pair must contain integers.'

def init_func():
    '''Called at the beginning of an animation.
    Credit: Alen Downey.
    '''
    pass

def animate_func(img):
    '''Draws one frame of the animation.
    Credit: Alen Downey.
    '''
    image.set_array(img)
    return (image,)

def draw(img, max_side_len=18):
    '''Draw an image with some handy defaults enabled.
    '''
    n, m = img.shape
    plt.axis([0, m, 0, n])
    options = dict(cmap='viridis') # tab20b
    options['extent'] = [0, m, 0, n]

    # plot
    im = plt.imshow(img, **options)

    # viz settings
    ax = plt.gca()
    fig = plt.gcf()

    if n >= m:
        xl = max_side_len; yl = max_side_len * (m / n)
    else:
        xl = max_side_len * (m / n); yl = max_side_len
    fig.set_size_inches(18, 5)

    plt.box(None)
    plt.grid()
    ax.set_xticks([])
    ax.set_yticks([])

    return im

# Main Classes #################################################################

class PetriDish:
    def __init__(self, shape, food=None, terrain=None, pheromone_evaporation=0.05):
        '''A petri dish in which slime agents will grow.

        Parameters
        ----------
        - shape (tuple):
            Tuple of integers.
        - food (np array):
            Two dimensional numpy array containing locations and amounts of food in PetriDish.
            Shape must match shape.
        - terrain (np array):
            Two dimensional numpy array containing terrain height values in PetriDish.
            Shape must match shape.
            NOT IMPLEMENTED.
        - pheromone_evaporation (float):
            Subtrahend for all non-zero pheromones at each step. Must be [0..1].
        '''
        # Set Shape
        check_tuple(shape, 'shape')
        self.shape = shape

        # Set Constants
        assert isinstance(pheromone_evaporation, float), 'pheromone_evaporation must be a float'
        assert pheromone_evaporation >= 0 and pheromone_evaporation <= 1, 'pheromone_evaporation must be [0..1]'
        self.pheromone_evaporation = pheromone_evaporation

        # Set Food and Terrain
        if food is not None:
            assert isinstance(food, np.ndarray), 'food must be a numpy array.'
            assert food.shape == self.shape, 'Shape of food must match shape of petri dish.'
            self.food = food

        if terrain is not None:
            assert isinstance(terrain, np.ndarray), 'terrain must be a numpy array.'
            assert food.shape == self.shape, 'Shape of terrain must match shape of petri dish.'
            self.terrain = terrain

        # Set Slime, Pheromones, and Edges for Later
        self.pheromones = np.zeros(shape)
        self.trails = np.zeros(shape)
        self.slime = np.zeros(shape)
        self.edges = np.pad(np.ones(shape), 1)

        # Empty list to record each step
        self.steps = []

    def grow_slime(self, agents):
        '''Run the slime mold simulation with agents.

        Parameters
        ----------
        - agents (list of Agents):
            Agents that will run around the petri dish.
        '''
        # TODO: Check types of agents and that they are properly initialized.
        n_steps = 0
        no_food = self.food.sum() == 0
        all_agents_home = all([a.is_home for a in agents])

        while not (no_food and all_agents_home):
            self.pheromones = np.where(self.pheromones > 0, self.pheromones - 0.025, 0) # new, to test
            current_locations = np.zeros(self.shape)
            for a in agents:
                a.make_next_move(self)
                x, y = a.loc
                current_locations[x, y] = 1
                if a.returning_home: current_locations[x, y] += 1

            # self.steps.append(current_locations)
            self.steps.append(np.copy(self.slime))
            n_steps += 1
            no_food = self.food.sum() == 0
            all_agents_home = all([a.is_home for a in agents])

        print(f'Finished after {n_steps}')
        print(f'There is no more food: {self.food.sum()}')
        print(f'All the agents are home: {all_agents_home}')

    # Setters

    def set_slime(self, p_loc, p_amt=1):
        '''Increment slime to say that an agent was here.

        Parameters
        ----------
        - p_loc (numpy array of shape 2,):
            Location of petri dish to increment slime.
        - p_amt (int):
            Amount of increment slime.
        '''
        check_np_pair(p_loc)
        assert isinstance(p_amt, int), 'p_amt must be int.'

        x, y = p_loc
        self.slime[x, y] += p_amt

    def set_pheromones(self, p_loc, p_amt=1):
        '''Increment pheromones to say that this is a good path towards food.

        Parameters
        ----------
        - p_loc (numpy array of shape 2,):
            Location of petri dish to increment pheromones.
        - p_amt (int):
            Amount of increment pheromones.
        '''
        check_np_pair(p_loc)
        assert isinstance(p_amt, int), 'p_amt must be int.'

        x, y = p_loc
        self.pheromones[x, y] += p_amt
        self.trails[x, y] += p_amt

    def set_food(self, p_loc):
        '''Decrement food at location p_loc.

        Parameters
        ----------
        - p_loc (numpy array of shape 2,):
            Location of petri dish to decrement food.
        '''
        check_np_pair(p_loc)

        x, y = p_loc
        self.food[x, y] -= 1

    # Visualization

    def animate(self, interval=50):
        '''Animate slime growing in the petri dish.

        Parameters
        ----------
        - interval (int):
            Time interval in milliseconds between animation frames.
        '''
        assert isinstance(interval, int), 'interval must be an integer.'
        fig = plt.gcf()
        # global image
        image = draw(self.steps[0])

        return FuncAnimation(fig, func=self.animate_func, init_func=init_func, frames=enumerate(self.steps), interval=interval)

    def animate_func(self, e):
        '''Draws one frame of the animation.
        Credit: Alen Downey.
        '''
        x, img = e
        if (x % 10) == 0: print(x)
        # image.set_array(self.steps[x])
        return plt.imshow(img)


class Agent:
    def __init__(self, shape, home=None, forward_bias=0.75, p_strength=5, view_distance=3):
        '''A slime agent for your petri dish.

        Parameters
        ----------
        - shape (tuple of shape 2,):
            Shape of the petri dish for slime agent.
        - home (optional, np array of shape 2,):
            Location of slime agent to start off at.
        - forward_bias (optional, float):
            Additional bias to assign current direction of slime.
        '''
        # Set Shape and Forward Bias
        check_tuple(shape, 'shape')
        self.shape = shape

        assert isinstance(forward_bias, float), 'foward bias must be a float.'
        self.forward_bias = forward_bias

        assert isinstance(p_strength, int), 'p_strength must be an int.'
        self.p_strength = p_strength

        assert isinstance(view_distance, int), 'view_distance must be an int.'
        self.view_distance = view_distance

        # Set Home
        if home is None:
            self.loc = np.array([np.random.randint(shape[0]), np.random.randint(shape[1])])
        else:
            check_np_pair(home)
            for h, s in zip(home, shape):
                assert h < s, f'Home must be within shape {shape}.'
            self.loc = np.array(home)

        self.home = np.copy(self.loc)

        # Set Random Direction
        self.direction = np.array([np.random.randint(-1, 2), np.random.randint(-1, 2)])

        while np.abs(self.direction).sum() < 1:
            self.direction = np.array([np.random.randint(-1, 2), np.random.randint(-1, 2)])

        # Attributes for Run Time
        self.returning_home = False
        self.is_home = True
        self.steps = []

    def make_next_move(self, petri):
        '''Make the agent's next move.

        Parameters
        ----------
        - petri (PetriDish object):
            The current PetriDish that the agent is running around in.
        '''
        if self.is_home and petri.food.sum() == 0:
            pass
        elif self.returning_home or petri.food.sum() == 0:
            self.semi_random_step_home(petri)
        else:
            xs, ys = self.get_field_of_view(petri)
            fov_foods = petri.food[xs[0]:xs[1], ys[0]:ys[1]]

            if np.any(fov_foods > 0):
                self.retreive_food(petri, xs, ys, fov_foods)
            else:
                self.make_exploratory_step(petri)

    def make_exploratory_step(self, petri):
        '''Make and exploratory step.

        Parameters
        ----------
        - petri (PetriDish object):
            The current PetriDish that the agent is running around in.
        '''
        dir_weights = self.semi_random_step_forward(petri, self.direction)
        self.direction = np.array(np.unravel_index(dir_weights.argmax(), dir_weights.shape)) - 1

        self.move(petri, set_pheromones=False)

    def semi_random_step_forward(self, petri, desired_direction):
        '''Return direction weights.

        Parameters
        ----------
        - petri (PetriDish object):
            The current PetriDish that the agent is running around in.
        - desired_direction (np array of shape (2,)):
            The desired direction that the agent should trend towards.
        '''
        assert petri.shape == self.shape, 'Shape of PetriDish does not match shape of SlimeAgent.'
        x, y = self.loc
        edges = petri.edges[x:x+3, y:y+3]

        phero = np.pad(petri.pheromones, 1)[x:x+3, y:y+3]
        if phero.sum() > 0:
            phero /= phero.sum()

        dir_weights = (np.random.uniform(size=(3, 3)) + phero * self.p_strength) * queen

        if edges.min() != 0:
            x, y = desired_direction + 1
            dir_weights[x, y] += self.forward_bias

            if np.abs(desired_direction).sum() == 2:
                rotation = (desired_direction * rotor).sum() + np.clip(desired_direction.sum(), 0, 2)
                dir_weights *= np.rot90(el, rotation)
            else:
                rotation = np.clip(desired_direction[0], 0, 1) + desired_direction[0] + desired_direction[1] * -1
                dir_weights *= np.rot90(row, rotation)
        else:
            dir_weights *= edges

        return dir_weights


    def next_step_home(self, petri):
        '''Make the fasted possible step home.

        Parameters
        ----------
        - petri (PetriDish object):
            The current PetriDish that the agent is running around in.
        '''
        self.direction = np.clip(self.home - self.loc, -1, 1)
        self.move(petri, set_pheromones=True, move_type='going home')

    def semi_random_step_home(self, petri):
        '''Take a semi-random step in rough direction of home.
        Testing.

        Parameters
        ----------
        - petri (PetriDish object):
            The current PetriDish that the agent is running around in.
        '''
        desired_direction = np.clip(self.home - self.loc, -1, 1)

        dir_weights = self.semi_random_step_forward(petri, desired_direction)
        self.direction = np.array(np.unravel_index(dir_weights.argmax(), dir_weights.shape)) - 1

        self.move(petri, set_pheromones=True, move_type='semi random home')

    def manhattan_step(self, petri, target):
        '''Take a manhattan step towards the target location.

        Parameters
        ----------
        - petri (PetriDish object):
            The current PetriDish that the agent is running around in.
        - target (x, y location on petri dish):
            The location that the slime is headed towards. For debugging purposes only.
        '''
        check_np_pair(target)
        self.direction = np.clip(target - self.loc, -1, 1)

        self.move(petri, set_pheromones=False, move_type='manhattan step', target=target)

    def get_field_of_view(self, petri):
        '''Get the agent's current field of view.

        Parameters
        ----------
        - petri (PetriDish object):
            The current PetriDish that the agent is running around in.
        '''
        return np.vstack((
            np.clip(self.loc - self.view_distance, 0, petri.shape[0]),
            np.clip(self.loc + self.view_distance + 1, 0, petri.shape[1]))).T

    def retreive_food(self, petri, xs, ys, fov_foods):
        '''Make a manhattan step towards one of the closest pieces of food.

        Parameters
        ----------
        - petri (PetriDish object):
            The current PetriDish that the agent is running around in.
        - xs & xy (each a tuple of integers):
            x's and y's for agent's current field of view.
        - fov_foods (np array):
            Food in field of view.
        '''
        ixs, iys = np.indices(petri.shape)[:, xs[0]:xs[1], ys[0]:ys[1]]
        food_locs = np.array((ixs[fov_foods > 0], iys[fov_foods > 0])).T
        target_food = food_locs[np.abs(food_locs - self.loc).sum(axis=1).argmin()]

        fx, fy = target_food
        assert petri.food[fx, fy] > 0, 'No food at target location.'

        while not np.all(self.loc == target_food):
            self.manhattan_step(
                petri,
                target_food)

    def move(self, petri, set_pheromones=False, move_type='exploratory', target=None):
        '''Move the agent one step based on current direction.
        Update agent and PetriDish attributes.

        Parameters
        ----------
        - petri (PetriDish object):
            The current PetriDish that the agent is running around in.
        - set_pheromones (bool):
            Whether or not to drop pheromones at the location we're moving to.
        - move_type (str):
            The type of move that is being made. For debugging purposes only.
        - target (x, y location on petri dish):
            The location that the slime is headed towards. For debugging purposes only.
        '''
        self.loc += self.direction
        self.steps.append({
            'loc':np.copy(self.loc),
            'direction':np.copy(self.direction),
            'move_type':move_type,
            'returning_home':self.returning_home,
            'is_home':self.is_home,
            'set_pheromones':set_pheromones,
            'target':target
            })

        petri.set_slime(self.loc)

        if set_pheromones:
            petri.set_pheromones(self.loc)

        x, y = self.loc

        if petri.food[x, y] > 0 and not self.returning_home:
            petri.set_food(self.loc)
            petri.set_pheromones(self.loc)
            self.returning_home = True

        self.is_home = (self.loc == self.home).all()

        if self.returning_home and self.is_home:
            self.returning_home = False
