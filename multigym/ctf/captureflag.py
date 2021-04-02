import math
import numpy as np

import labmaze
import gym_minigrid.minigrid as minigrid
from gym_minigrid import rendering

import multigym.multigrid as multigrid

from multigym.register import register


class TeamBase(minigrid.WorldObj):
    def __init__(self, color='red'):
        super(TeamBase, self).__init__(color)

    def can_overlap(self):
        return True

    def render(self, img):
        c = minigrid.COLORS[self.color]
        # Vertical quad

        minigrid.fill_coords(img, minigrid.point_in_rect(0.20, 0.80, 0.90, 0.95), c)

    def can_pickup(self):
        return False


class TeamAgent(multigrid.Agent):
    def __init__(self, agent_id, state, team_color):
        super().__init__(agent_id, state)
        self.team_color = team_color

    def render(self, img):
        tri_fn = rendering.point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        tri_fn = rendering.rotate_fn(
            tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        c = minigrid.COLORS[self.team_color]
        rendering.fill_coords(img, tri_fn, c)


class Flag(minigrid.WorldObj):

    def __init__(self, color='red'):
        super(Flag, self).__init__(type='goal', color=color)

    def can_overlap(self):
        return True

    def render(self, img):
        c = minigrid.COLORS[self.color]
        # Vertical quad
        minigrid.fill_coords(img, minigrid.point_in_rect(0.35, 0.45, 0.31, 0.88), c)
        minigrid.fill_coords(img, minigrid.point_in_triangle(
            (0.35, 0.31),
            (0.80, 0.50),
            (0.35, 0.60),
        ), c)

    def can_pickup(self):
        return True


class CapturingTheFlag(multigrid.MultiGridEnv):

    def __init__(self,
                 teams,
                 players_per_team,
                 scores_to_win,
                 grid_size=None,
                 width=None,
                 height=None,
                 max_steps=100,
                 see_through_walls=False,
                 seed=1234,
                 agent_view_size=7,
             ):
        """

        Args:
            teams: number of teams in the game
            agents_per_team: an integer or a list of number of agents per team
            scores_to_win: number of capture-return events to consider a game completed
            grid_size:
            width:
            height:
            max_steps:
            see_through_walls:
            seed:
            agent_view_size:
        """
        self.num_teams = teams
        self.players_per_team = players_per_team
        self.scores_to_win = scores_to_win

        self.base_arena = labmaze.RandomMaze(
            width=width,
            height=height,
            max_rooms=teams,
            spawns_per_room=players_per_team,
            objects_per_room=1,
            random_seed=seed,
            simplify=False,
        )
        super().__init__(
            grid_size=grid_size,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            seed=seed,
            agent_view_size=agent_view_size,
            n_agents=teams * players_per_team,
            competitive=True,
            fixed_environment=False,
            minigrid_mode=False,
            fully_observed=False
        )

    def _gen_grid(self, width, height):
        self.grid = multigrid.Grid(height=height, width=width)
        for i in range(width):
            for j in range(height):
                entry = self.base_arena.entity_layer[i, j]
                if entry == '*':
                    self.put_obj(minigrid.Wall(), i, j)

        self.teams_colors = list(sorted(minigrid.COLORS.keys()))
        self.flags_pos = np.argwhere(self.base_arena.entity_layer == self.base_arena.object_token)
        for team in range(self.num_teams):
            team_flag = Flag(color=self.teams_colors[team])
            self.put_obj(
                team_flag,
                self.flags_pos[team][0],
                self.flags_pos[team][1]
            )

        self.place_agent()
        self.mission = "capture the opponent's flag"

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
        self.spawn_points = np.argwhere(self.base_arena.entity_layer == self.base_arena.spawn_token)
        self.teams = []

        for i in range(self.n_agents):
            team_preferences = self._get_team_preference_for_player(self.spawn_points[i])
            team = team_preferences[0]
            self.place_agent_at_pos(
                i,
                self.spawn_points[i],
                agent_obj=TeamAgent(
                    agent_id=i,
                    state=self._rand_int(0, 4),
                    team_color=self.teams_colors[team]
                )
            )

    def _get_team_preference_for_player(self, agent_pos):
        return np.argsort(np.sum(np.abs(self.flags_pos - agent_pos), axis=1))  # order by Manhattan (Taxicab) distance

    def step_one_agent(self, action, agent_id):
        raise NotImplementedError()

    def move_agent(self, agent_id, new_pos):
        raise NotImplementedError()


class CaptureFlagClassicEnv(CapturingTheFlag):
    def __init__(self):
        super().__init__(
            teams=2,
            players_per_team=2,
            scores_to_win=2,
            height=19,
            width=19
        )



if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
    env_id='MultiGrid-CTF-Classic-v0',
    entry_point=module_path + ':CaptureFlagClassicEnv'
)