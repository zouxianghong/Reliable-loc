#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: some functions for MCL initialization

import os
import numpy as np

np.random.seed(0)


def init_particles_uniform(map_size, numParticles, init_weight=1.0):
    """ Initialize particles uniformly.
        Args:
            map_size: size of the map.
            numParticles: number of particles.
        Return:
            particles.
    """
    [x_min, x_max, y_min, y_max] = map_size
    particles = []
    rand = np.random.rand
    for i in range(numParticles):
        x = (x_max - x_min) * rand(1) + x_min
        y = (y_max - y_min) * rand(1) + y_min
        z = 0.0
        # theta = 2 * np.pi * rand(1)
        theta = -np.pi + 2 * np.pi * rand(1)
        particles.append([x, y, z, theta, init_weight])

    return np.array(particles)


def init_particles_given_coords(numParticles, coords, init_weight=1.0):
    """ Initialize particles uniformly given the road coordinates.
        Args:
            numParticles: number of particles.
            coords: road coordinates
        Return:
            particles.
    """
    particles = []
    rand = np.random.rand
    args_coords = np.arange(len(coords))
    selected_args = np.random.choice(args_coords, numParticles)

    for i in range(numParticles):
        x = coords[selected_args[i], 0]
        y = coords[selected_args[i], 1]
        z = 0.0
        theta = -np.pi + 2 * np.pi * rand(1)[0]
        particles.append([x, y, z, theta, init_weight])

    return np.array(particles, dtype=float)


def get_coords(dataset, map_trip_idx):
    """ Compute the size of the map.
        Args:

        Return:
            the road coordinates.
    """
    grid_coords = []
    indices = dataset.dataset.get_indices_in_dataset()
    map_indices = indices[map_trip_idx]
    for idx in map_indices:
        coord_xy = dataset.dataset.get_pos_xy(idx)
        grid_coords.append(coord_xy)
    grid_coords = np.array(grid_coords, dtype=float)
    # map size
    min_x = np.min(grid_coords[:, 0])
    max_x = np.max(grid_coords[:, 0])
    min_y = np.min(grid_coords[:, 1])
    max_y = np.max(grid_coords[:, 1])
    return [min_x, max_x, min_y, max_y], grid_coords
