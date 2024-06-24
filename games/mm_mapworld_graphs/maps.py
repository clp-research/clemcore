# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx

from glob import glob
import json
import gzip
import os
import matplotlib.pyplot as plt


class AbstractMap(object):
    dir2delta = {'n': np.array((-1, 0)),
                 's': np.array((1, 0)),
                 'e': np.array((0, 1)),
                 'w': np.array((0, -1))}

    def __init__(self, n, m, n_rooms):
        if n*m < n_rooms:
            raise ValueError('n*m must be larger than n_rooms')
        self.n = n
        self.m = m
        self.n_rooms = n_rooms
        self.G = self.make_graph(n, m, n_rooms)

    def make_graph(self, n, m, n_rooms):
        map_array = np.zeros((n, m))
        G = nx.Graph()

        current_pos = np.random.randint(0, n), np.random.randint(0, m)
        map_array[current_pos] = 1
        G.add_node(current_pos)

        while map_array.sum() < n_rooms:
            random_dir = np.random.choice(['n', 's', 'e', 'w'])
            new_pos = tuple(np.array(current_pos) + self.dir2delta[random_dir])
            if min(new_pos) < 0 or new_pos[0] >= n or new_pos[1] >= m:
                # illegal move
                continue
            map_array[new_pos] = 1
            G.add_node(new_pos)
            G.add_edge(current_pos, new_pos)
            current_pos = new_pos
        return G

    def plot_graph(self):
        nx.draw_networkx(self.G, pos={n: n for n in self.G.nodes()})

    def __repr__(self):
        return '<AbstractMap({}, {}, {})>'.format(self.n, self.m, self.n_rooms)