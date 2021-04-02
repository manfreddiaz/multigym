import numpy as np
import labmaze as lbm


def create_random_maze():
    return lbm.RandomMaze(
        height=15,
        width=15,
        max_rooms=2, # number of teams
        room_min_size=4,
        room_max_size=4,
        spawns_per_room=2,
        objects_per_room=1,
        random_seed=42,
        simplify=False,
        extra_connection_probability=0.9
    )


def test_random_generation():
    maze = create_random_maze()
    for i in range(100):
        print(maze.entity_layer)
        maze.regenerate()


def test_random_generation_from_base():
    maze = create_random_maze()
    el = maze.entity_layer
    el[el == 'P'] = ' '
    fixed_maze = lbm.FixedMazeWithRandomGoals(
        str(el),
        num_spawns=4,
        num_objects=2
    )
    for i in range(100):
        entity_layer = fixed_maze.entity_layer
        print(entity_layer)
        print(np.argwhere(entity_layer == 'G'))
        fixed_maze.regenerate()


# test_random_generation()
test_random_generation_from_base()
