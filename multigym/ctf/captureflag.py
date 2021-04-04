import math
from enum import IntEnum

import numpy as np
import labmaze

from gym_minigrid import minigrid
from multigym import multigrid, register
from .objects import TeamAgent, Flag, Ray

EVENTS = {
    'flag_capture': 6.0,
    'flag_pickup': 1.0,
    'flag_return': 1.0,
    'flag_teammate': 5.0,
    'tag_with_flag': 2.0,
    'tag_without_flag': 1.0
}


class CapturingTheFlag(multigrid.MultiGridEnv):

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        # Pick up an object
        pickup = 3
        # Drop an object
        drop = 4
        # Toggle/activate an object
        toggle = 5
        # tag another player
        tag = 6
        # do nothing
        no_op = 7

    def __init__(self,
                 teams,
                 players_per_team,
                 scores_to_win,
                 player_health=3,
                 player_respawn=5,
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
            agents_per_team: number of agents per team
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
        self.player_health = player_health
        self.player_respawn = player_respawn

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
            competitive=False,
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
            team_flag = Flag(
                team_id=team,
                color=self.teams_colors[team]
            )
            self.put_obj(
                team_flag,
                self.flags_pos[team][0],
                self.flags_pos[team][1]
            )

        self.place_agent()
        self.actions = CapturingTheFlag.Actions
        self.mission = "capture the opponent's flag"

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
        self.spawn_points = np.argwhere(self.base_arena.entity_layer == self.base_arena.spawn_token)
        self.teams = {}
        self.beams = {}
        self.health = {}
        self.respawn = {}

        for agent_id in range(self.n_agents):
            team_preferences = self._get_team_preference_for_player(self.spawn_points[agent_id])
            team = team_preferences[0]  # TODO: Improve team selection
            self.teams[agent_id] = team
            agent_obj = TeamAgent(
                agent_id=agent_id,
                state=self._rand_int(0, 4),
                team_id=team,
                team_color=self.teams_colors[team]
            )
            self.place_agent_at_pos(
                agent_id,
                self.spawn_points[agent_id],
                agent_obj=agent_obj
            )
            agent_obj.init_pos = self.spawn_points[agent_id]
            self.health[agent_id] = self.player_health

    def _get_team_preference_for_player(self, agent_pos):
        return np.argsort(np.sum(np.abs(self.flags_pos - agent_pos), axis=1))  # order by Manhattan (Taxicab) distance

    def _pickup(self, agent_id, fwd_pos):
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell and fwd_cell.can_pickup():
            if isinstance(fwd_cell, Flag):
                if fwd_cell.team_id == self.teams[agent_id] and not np.array_equal(fwd_cell.cur_pos, fwd_cell.init_pos):
                    # verify that the cell of initial pos is emtpy
                    if not self.grid.get(*fwd_cell.init_pos):
                        self.grid.set(fwd_cell.cur_pos[0], fwd_cell.cur_pos[1], None)
                        np.copyto(fwd_cell.cur_pos, fwd_cell.init_pos)
                        self.grid.set(fwd_cell.cur_pos[0], fwd_cell.cur_pos[1], fwd_cell)
                        return fwd_cell
                elif fwd_cell.team_id != self.teams[agent_id]:
                    super(CapturingTheFlag, self)._pickup(agent_id, fwd_pos)

        return None

    def _forward(self, agent_id, fwd_pos):
        """Attempts to move the forward one cell, returns True if successful."""
        fwd_cell = self.grid.get(*fwd_pos)
        # Make sure agents can't walk into each other
        agent_blocking = False
        for a in range(self.n_agents):
            if a != agent_id and np.array_equal(self.agent_pos[a], fwd_pos):
                agent_blocking = True

        # Deal with object interactions
        if not agent_blocking and fwd_cell is None or fwd_cell.can_overlap():
            self.move_agent(agent_id, fwd_pos)
            return True

        return False

    def _drop(self, agent_id, fwd_pos):
        dropped = super(CapturingTheFlag, self)._drop(agent_id, fwd_pos)
        if not dropped and self.carrying[agent_id]:  # couldn't drop, agent is carrying
            fwd_cell = self.grid.get(*fwd_pos)
            # verify if forward cell is the agent's flag and it's on the base
            if isinstance(fwd_cell, Flag) and fwd_cell.team_id == self.teams[agent_id] and fwd_cell.cur_pos == fwd_cell.init_pos:
                # return carrying flag to other team's base
                carrying_flag = self.carrying[agent_id]
                np.copyto(carrying_flag.cur_pos, carrying_flag.init_pos)
                self.grid.set(carrying_flag.cur_pos[0], carrying_flag.cur_pos[1], carrying_flag)
                # the agent is no longer carrying the other team's flag
                self.carrying[agent_id] = None
                agent_pos = self.agent_pos[agent_id]
                agent = self.grid.get(*agent_pos)
                agent.contains = None
                return True

        return False

    def _ray_cast(self, agent_id):
        ray = []

        ray_pos = self.front_pos[agent_id]
        fwd_cell = self.grid.get(*ray_pos)
        while not fwd_cell:
            ray.append(ray_pos)
            ray_pos = ray_pos + self.dir_vec[agent_id]
            fwd_cell = self.grid.get(*ray_pos)

        return ray, fwd_cell

    def _tag(self, agent_id, fwd_pos):
        # can't tag if it is carrying the flag
        if self.carrying[agent_id]:
            return False

        ray, tagged_object = self._ray_cast(agent_id)
        for position in ray:
            self.grid.set(position[0], position[1], Ray(self.teams_colors[self.teams[agent_id]]))
        self.beams[agent_id] = ray

        return tagged_object

    def step_one_agent(self, action, agent_id):
        reward = 0

        if agent_id in self.respawn:
            if self.respawn[agent_id]['count'] > 0:
                # TODO(manfred) receive collective reward.
                self.respawn[agent_id]['count'] -= 1
                return reward
            else:
                agent_info = self.respawn.pop(agent_id)
                agent_pos = self.agent_pos[agent_id]
                self.grid.set(agent_pos[0], agent_pos[1], agent_info['agent'])
                # TODO(manfred) solve bug that when respawning an agent there may be another object on that position
                # this is solvable by placing the agent on the spawning places of the team

        # Get the position in front of the agent
        fwd_pos = self.front_pos[agent_id]

        # remove the beam from previous timestep
        if agent_id in self.beams and isinstance(self.beams[agent_id], list):
            ray = self.beams[agent_id]
            for cell in ray:
                self.grid.set(cell[0], cell[1], None)
            self.beams.pop(agent_id)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir[agent_id] -= 1
            if self.agent_dir[agent_id] < 0:
                self.agent_dir[agent_id] += 4
            self.rotate_agent(agent_id)

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir[agent_id] = (self.agent_dir[agent_id] + 1) % 4
            self.rotate_agent(agent_id)

        # Move forward
        elif action == self.actions.forward:
            self._forward(agent_id, fwd_pos)

        # Pick up an object
        elif action == self.actions.pickup:
            picked_flag = self._pickup(agent_id, fwd_pos)
            if picked_flag:
                if picked_flag.team_id == self.teams[agent_id]:
                    reward += EVENTS['flag_return']
                else:
                    reward += EVENTS['flag_pickup']

        # Drop an object
        elif action == self.actions.drop:
            if self._drop(agent_id, fwd_pos):
                reward += EVENTS['flag_capture']

        # Toggle/activate an object
        elif action == self.actions.toggle:
            toggle = self._toggle(agent_id, fwd_pos)

        elif action == self.actions.tag:
            tagged_object = self._tag(agent_id, fwd_pos)
            if isinstance(tagged_object, TeamAgent) and tagged_object.team_id != self.teams[agent_id]:
                tagged_agent: TeamAgent = tagged_object

                self.health[tagged_agent.agent_id] -= 1
                # TODO(manfred) interesting problem here, there is a difference in reward structure
                # one interpretation is to give reward for every time the agent tags another with / without flag
                # or only when the tag effectively returns the flag
                # a possible extension is for multiple teams: give reward for tagging even if the flag is another team
                if self.carrying[tagged_agent.agent_id]:
                    if tagged_agent.contains.team_id == self.teams[agent_id]:  # if the tagged agent had my team's flag
                        reward += EVENTS['tag_with_flag']
                    else:
                        reward += EVENTS['tag_without_flag']  # tag if has the flag of another team
                else:
                    reward += EVENTS['tag_without_flag']
                if self.health[tagged_agent.agent_id] == 0:
                    # remove player from grid
                    self.grid.set(tagged_agent.cur_pos[0], tagged_agent.cur_pos[1], None)

                    if self.carrying[tagged_agent.agent_id]:
                        flag = self.carrying[tagged_agent.agent_id]
                        np.copyto(flag.cur_pos, tagged_agent.cur_pos)
                        self.grid.set(flag.cur_pos[0], flag.cur_pos[1], flag)
                        self.carrying[tagged_agent.agent_id] = None
                        tagged_agent.contains = None
                    # TODO(manfred) after being dropped the flag disappears
                    # TODO(manfred) test respawn and flag dropping
                    # player's inactive
                    self.respawn[tagged_agent.agent_id] = {
                        'count': self.player_respawn,
                        'agent': tagged_agent
                    }
                    np.copyto(self.agent_pos[tagged_agent.agent_id], tagged_agent.init_pos)
                    self.health[tagged_agent.agent_id] = self.player_health


        # Done action -- by default acts as no-op.
        elif action == self.actions.no_op:
            pass
        else:
            assert False, 'unknown action'

        return reward

    # def step(self, actions):
    #     raise NotImplementedError()



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