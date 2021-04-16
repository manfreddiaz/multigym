import math
from enum import IntEnum

import numpy as np
import labmaze

from gym_minigrid import minigrid
from multigym import multigrid, register

from .objects import Player, Team, Flag, RespawnPool, Beam
from .arena import ArenaGenerator

REWARDS = {
    'flag_capture': 6.0,
    'flag_pickup': 1.0,
    'flag_return': 1.0,
    'flag_teammate': 5.0,
    'tag_with_flag': 2.0,
    'tag_without_flag': 1.0,
    'invalid_action': -1.0,
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
                 scores_to_win,
                 player_health=3,
                 player_respawn=3,
                 width=None,
                 height=None,
                 max_steps=500,
                 see_through_walls=False,
                 seed=34,
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
        self.scores_to_win = scores_to_win
        self.player_health = player_health
        self.player_respawn = player_respawn

        self.base_arena = ArenaGenerator(seed=seed)
        self.listeners = []

        super().__init__(
            grid_size=None,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=see_through_walls,
            seed=seed,
            agent_view_size=agent_view_size,
            n_agents=4,
            competitive=False,
            fixed_environment=False,
            minigrid_mode=False,
            fully_observed=False
        )

    def _get_actions(self):
        return CaptureFlagClassicEnv.Actions

    def _gen_grid(self, width, height):
        self.players = []
        self.respawn_pool = RespawnPool()
        self.beam_collection = []

        arena = self.base_arena.regenerate()
        self.height, self.width = arena.shape
        self.grid = multigrid.Grid(width=self.width, height=self.height)

        for i in range(self.width):
            for j in range(self.height):
                entry = arena[j, i]
                if entry == '*':
                    self.put_obj(minigrid.Wall(), i, j)

        for team in self.base_arena.teams:
            self.put_obj(
                team,
                team.flag.init_pos[0],  # Coordinates of numpy and gym_minigrid are inverted
                team.flag.init_pos[1]
            )
            team.flag.cur_pos = team.cur_pos

        self.place_agent()
        self.actions = CapturingTheFlag.Actions
        self.mission = "capture the opponent's flag"

    def _respawn(self, player):
        re_spawned = False

        respawn_points = player.team.respawns
        spawn_point = None
        while not re_spawned:  # TODO(manfred): although it is unlikely that the agent can't be placed back
            index = self._rand_int(0, len(respawn_points))
            re_spawned = self.grid.get(respawn_points[index][0], respawn_points[index][1]) is None
            spawn_point = respawn_points[index] if re_spawned else None

        if spawn_point is not None:
            self.place_agent_at_pos(
                player.agent_id,
                spawn_point,
                agent_obj=player
            )
            player.init_pos = spawn_point

        player.respawn()

        return spawn_point

    def _emit(self, event, agent_id):
        for listener in self.listeners:
            listener.emit(event, agent_id)

        return REWARDS[event]

    def listen(self, listener):
        assert hasattr(listener, 'emit')
        self.listeners.append(listener)

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
        for agent_id in range(self.n_agents):
            team_id = agent_id // 2
            team = self.base_arena.teams[team_id]
            player = Player(
                agent_id=agent_id,
                state=self._rand_int(0, 4),
                health=self.player_health,
                respawn_after=self.player_respawn
            )
            team.add_player(player)
            self._respawn(player)
            self.players.append(player)

    def _pickup(self, agent_id, fwd_pos):
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell and fwd_cell.can_pickup():
            flag = None
            if isinstance(fwd_cell, Team):
                flag = fwd_cell.flag
            elif isinstance(fwd_cell, Flag):
                flag = fwd_cell

            if flag:
                player = self.players[agent_id]
                picked_up = flag.pick_up(player)
                returned = flag.pick_return(player)
                if picked_up or returned:
                    if isinstance(fwd_cell, Flag):
                        self.grid.set(fwd_pos[0], fwd_pos[1], None)

                if picked_up:
                    return self._emit('flag_pickup', agent_id)
                if returned:
                    return self._emit('flag_return', agent_id)

        return 0.0

    def _forward(self, agent_id, fwd_pos):
        """Attempts to move the forward one cell, returns True if successful."""
        fwd_cell = self.grid.get(*fwd_pos)
        # Make sure agents can't walk into each other
        agent_blocking = False
        for a in range(self.n_agents):
            if a != agent_id and np.array_equal(self.agent_pos[a], fwd_pos):
                agent_blocking = True

        # Deal with object interactions
        if not agent_blocking and fwd_cell is None:
            self.move_agent(agent_id, fwd_pos)
            return 0.0

        return self._emit('invalid_action', agent_id)

    def _drop(self, agent_id, fwd_pos):
        player = self.players[agent_id]
        if player.is_holding:
            flag = player.holding
            fwd_cell = self.grid.get(*fwd_pos)
            if not fwd_cell:
                player.drop()
                self.grid.set(fwd_pos[0], fwd_pos[1], flag)
                flag.cur_pos = fwd_pos
                return False
            elif isinstance(fwd_cell, Team):
                team: Team = fwd_cell

                if team.id == player.team.id and isinstance(flag, Flag) and not team.flag.is_held:
                    player.drop()
                    flag.returns()
                    return self._emit('flag_capture', agent_id)
                else:
                    player.drop()
                    flag.returns()

        return 0.0

    def _beam(self, agent_id, fwd_pos):
        beam = []

        pos = fwd_pos
        fwd_cell = self.grid.get(*pos)
        while not fwd_cell:
            beam.append(pos)
            pos = pos + self.dir_vec[agent_id]
            fwd_cell = self.grid.get(*pos)

        return beam, fwd_cell

    def _tag(self, agent_id, fwd_pos):
        player = self.players[agent_id]

        if player.is_holding:
            return None

        beam, tagged = self._beam(agent_id, fwd_pos)
        self.beam_collection.append((beam, player))

        if tagged and isinstance(tagged, Player):
            is_tagged = player.tag(tagged)
            is_holding = tagged.is_holding
            if not tagged.health:
                self.grid.set(tagged.cur_pos[0], tagged.cur_pos[1], None)
                self.respawn_pool.add_player(tagged)
                if is_holding:
                    self._drop(tagged.agent_id, tagged.cur_pos)

            if is_tagged and is_holding:
                return self._emit('tag_with_flag', agent_id)
            elif is_tagged and not is_holding:
                return self._emit('tag_without_flag', agent_id)

        return 0.0

    def step_one_agent(self, action, agent_id):
        reward = 0.0

        if not self.players[agent_id].active:
            return reward

        # Get the position in front of the agent
        fwd_pos = self.front_pos[agent_id]

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
            reward += self._forward(agent_id, fwd_pos)

        # Pick up an object
        elif action == self.actions.pickup:
            reward += self._pickup(agent_id, fwd_pos)

        # Drop an object
        elif action == self.actions.drop:
            reward += self._drop(agent_id, fwd_pos)

        # Toggle/activate an object
        elif action == self.actions.toggle:
            toggle = self._toggle(agent_id, fwd_pos)
            reward += -1 if not toggle else 0.0

        elif action == self.actions.tag:
            reward += self._tag(agent_id, fwd_pos)

        # Done action -- by default acts as no-op.
        elif action == self.actions.no_op:
            pass
        else:
            assert False, 'unknown action'

        return reward

    def step(self, actions):
        respawned = self.respawn_pool.tick()
        for player in respawned:
            self._respawn(player)

        for beam, player in self.beam_collection:
            for position in beam:
                fwd_cell = self.grid.get(*position)
                if fwd_cell and isinstance(fwd_cell, Beam):
                    self.grid.set(position[0], position[1], None)

        self.beam_collection.clear()

        returned = super(CapturingTheFlag, self).step(actions)

        for beam, player in self.beam_collection:
            for position in beam:
                fwd_cell = self.grid.get(*position)
                if fwd_cell is None:
                    self.grid.set(position[0], position[1], Beam(player.team.color))

        return returned


class CaptureFlagClassicEnv(CapturingTheFlag):
    def __init__(self, seed=34):
        super().__init__(
            scores_to_win=2,
            height=13,
            width=13,
            seed=seed
        )

if hasattr(__loader__, 'name'):
  module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
  module_path = __loader__.fullname

register(
    env_id='MultiGrid-CTF-Classic-v0',
    entry_point=module_path + ':CaptureFlagClassicEnv'
)