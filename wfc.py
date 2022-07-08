from mercantile import neighbors
import networkx as nx
import numpy as np

from dataclasses import dataclass


@dataclass
class Tile:
    """Tile data for generating terrain with a WFC alg."""
    name: str
    color: str
    draw: object = None

    def __hash__(self) -> int:
        return hash(repr(self.name))

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Tile):
            return self.name == __o.name
        return False

    def __lt__(self, __o):
        assert isinstance(__o, Tile), 'can only compare tiles with tiles'
        return self.name < __o.name


class WaveMap:
    """"""
    def __init__(self, x, y, adj) -> None:
        self.x = x
        self.y = y
        self.adj = adj
        self.shape = (x, y)

        self.G = get_graph(x, y)

    def __len__(self):
        return self.x * self.y

    def set_options(self, loc, options):
        self.G.nodes[loc]['options'] = set(options)
        self.G.nodes[loc]['n'] = len(options)

    def collapse(self, loc, options):
        self.set_options(loc, options)

        neigh_options = np.argwhere(
            self.adj[options, :].sum(axis=0) == len(options)
            ).ravel()

        self.update_neighbors(loc, neigh_options)

    def update_neighbors(self, loc, options):
        to_update = self.G.neighbors(loc)
        to_update = [n for n in to_update if self.G.nodes[n]['n'] != 1]
        
        for neigh in to_update:
            neighos = self.G.nodes[neigh]['options']
            if len(neighos) == 0:
                new_options = options
            else:
                new_options = [o for o in neighos if o in options]
                assert len(new_options) > 0, 'there must be at least one option' # self.rollback()

            if len(new_options) == 1:
                self.collapse(neigh, new_options)
            else:
                self.set_options(neigh, new_options)

    def rollback(self):
        # TODO implement and remove assertion requiring at least one option, above
        pass

    def get_ns(self):
        return np.array([self.G.nodes[x]['n'] for x in self.G.nodes])

    def run(self):
        def get_next(ns, high):
            if (ns == 0).all():
                return np.random.randint(high)
            min_n = ns[ns > 1].min()
            return np.random.choice(np.argwhere(ns == min_n).ravel())

        def select_options(self, loc):
            node = self.G.nodes[loc]
            if node['n'] == 0:
                return np.random.randint(len(self.adj))
            return np.random.choice(list(node['options']))

        ns = self.get_ns()
        i = 0
        
        while not (ns == 1).all():
            next_loc = get_next(ns, self.x * self.y)
            choice = select_options(self, next_loc)
            self.collapse(next_loc, [choice])

            ns = self.get_ns()
            i += 1

        print(f'completed in {i} steps')


def get_graph(x, y):
    G = nx.Graph()

    for i in range(x * y):
        G.add_node(i, options=[], n=0)

        if valid_loc(i - 1, (x, y)) and i % y != 0:
            G.add_edge(i, i - 1)
        if valid_loc(i - y, (x, y)):
            G.add_edge(i, i - y)

    return G


def valid_loc(loc, shape):
    x, y = shape

    return loc >= 0 and loc < (x * y)