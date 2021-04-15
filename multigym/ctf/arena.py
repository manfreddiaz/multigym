import queue
from math import ceil

import numpy as np
import labmaze as lbm
from gym_minigrid import minigrid
from multigym.ctf.objects import Team


RND_WIDTH = np.arange(15, 25 + 1, 2)
RND_HEIGHT = np.arange(9, 15 + 1, 2)
MAX_ROOMS = 10
SPAWNS_PER_ROOM = 3


def distance_from(grid, init_pos, end_pos=None):
    q = queue.Queue()
    q.put(init_pos)

    distances = np.ones(shape=grid.shape, dtype=np.float32) * np.infty
    distances[init_pos[0], init_pos[1]] = 0

    found = False
    while not q.empty() and not found:
        pos = q.get()
        for direction in minigrid.DIR_TO_VEC:
            next_pos = pos + direction
            if 0 <= next_pos[0] < grid.shape[0] and 0 <= next_pos[1] < grid.shape[1]:
                if grid[next_pos[0], next_pos[1]] != '*':
                    if distances[next_pos[0], next_pos[1]] > distances[pos[0], pos[1]] + 1:
                        q.put(next_pos)
                        distances[next_pos[0], next_pos[1]] = distances[pos[0], pos[1]] + 1
                    if np.array_equal(next_pos, end_pos):
                        found = True

    return distances


def array_to_textgrid(array):
    str_rpr = []
    for line in array:
        for c in line:
            str_rpr.append(c)
        str_rpr.append('\n')
    return lbm.TextGrid(''.join(str_rpr))


class ArenaGenerator:

    def __init__(self, num_spawn=3, seed=1234):
        self.red_team = None
        self.blue_team = None
        self.grid = None
        self.num_spawn = num_spawn
        self.seed_vale = seed
        self.seed(seed)
        self.regenerate()

    @property
    def teams(self):
        return [self.red_team, self.blue_team]

    @property
    def players(self):
        return [player for team in self.teams for player in team.players]

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def _create_grid(self):
        maze = lbm.RandomMaze(
            width=self.rng.choice(RND_WIDTH),
            height=self.rng.choice(RND_HEIGHT),
            max_rooms=MAX_ROOMS,
            spawns_per_room=0,
            objects_per_room=0,
            random_seed=self.rng.randint(2147483648),
            simplify=True,
        )
        red_base = maze.entity_layer[:, :ceil(maze.width // 2)]
        red_team = Team(
            team_id=0,
            team_color='red'
        )

        self._fill_team(red_team, red_base, np.array([0, 0]))

        blue_base = np.rot90(red_base, 2)
        blue_team = Team(
            team_id=1,
            team_color='blue'
        )
        self._fill_team(blue_team, blue_base, np.array([0, red_base.shape[1]]))

        arena = np.hstack((red_base, blue_base))

        return arena, red_team, blue_team

    def regenerate(self, max_tries=20):
        valid_arena = False
        tries = 0
        while not valid_arena and tries < max_tries:
            arena, red_team, blue_team = self._create_grid()
            valid_arena = self._is_valid(arena, red_team, blue_team)
            tries += 1

        if not valid_arena:
            return None

        self.grid = array_to_textgrid(arena)
        self.red_team = red_team
        self.blue_team = blue_team

        return self.grid

    def _fill_team(self, team, base, offset):
        if np.array_equal(offset, [0, 0]):
            area = np.argwhere(base == ' ')
        else:
            area = np.argwhere(base == ' ')[::-1]  # for the blue base, the empty space is ordered in reverse

        team.flag.init_pos = (area[0] + offset)[::-1]  # to match gym_minigrid coordinate system

        distances = distance_from(base, area[0])  # compute the closest points to the flag position

        respawn_candidates = np.array(np.unravel_index(np.argsort(distances, axis=None), distances.shape)).T[1:]
        for i in range(SPAWNS_PER_ROOM):
            team.add_respawn((respawn_candidates[i] + offset)[::-1])
        return team

    def __str__(self):
        str_rpr = []
        for i, line in enumerate(self.grid):
            for j, c in enumerate(line):
                indexes = np.array([i, j])
                if np.array_equal(self.red_team.flag.init_pos, indexes):
                    str_rpr.append('R')
                elif np.array_equal(self.blue_team.flag.init_pos, indexes):
                    str_rpr.append('B')
                # elif self.red_team.base.respawn.index(indexes) or self.blue_team.base.respawn.index(indexes):
                #     str_rpr.append('P')
                else:
                    str_rpr.append(c)
            str_rpr.append('\n')
        return ''.join(str_rpr)

    def _is_valid(self, arena, red_team, blue_team):
        distances = distance_from(arena, red_team.flag.init_pos, blue_team.flag.init_pos)
        distance_from_bases = distances[blue_team.flag.init_pos[1], blue_team.flag.init_pos[0]]
        return distance_from_bases != np.infty and distance_from_bases > 6



