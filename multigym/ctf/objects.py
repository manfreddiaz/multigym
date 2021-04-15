import math
import numpy as np
from gym_minigrid import minigrid, rendering
from multigym import multigrid, WorldObj, Agent


class Beam(WorldObj):
    def __init__(self, color):
        super().__init__('ball', color)

    def render(self, img):
        minigrid.fill_coords(
            img,
            minigrid.point_in_circle(0.5, 0.5, 0.31),
            minigrid.COLORS[self.color]
        )


class Flag(WorldObj):

    def __init__(self, team):
        super().__init__('goal', team.color)
        self._team = team
        self._holder = None

    def pick_up(self, player):
        can_pickup = player.team.id != self._team.id and not self._holder
        if can_pickup:
            self._holder = player
            player.hold(self)
            self.cur_pos = player.cur_pos

        return can_pickup

    def pick_return(self, player):
        returned = player.team.id == self._team.id and not np.array_equal(self.cur_pos, self.init_pos)
        if returned:
            self.returns()
        return returned

    def returns(self):
        self.cur_pos = self.init_pos

    def drop(self, player):
        assert self._holder == player
        self._holder = None

    @property
    def is_held(self):
        return self._holder is not None or not np.array_equal(self.cur_pos, self.init_pos)

    @property
    def team(self):
        return self._team

    def render(self, img):
        c = minigrid.COLORS[self.team.color]
        # Vertical quad
        minigrid.fill_coords(img, minigrid.point_in_rect(0.35, 0.45, 0.31, 0.88), c)
        minigrid.fill_coords(img, minigrid.point_in_triangle(
            (0.35, 0.31),
            (0.80, 0.50),
            (0.35, 0.60),
        ), c)

    def can_pickup(self):
        return True


class Team(WorldObj):

    def __init__(self, team_id, team_color):
        super().__init__('goal')
        self._id = team_id
        self._players = []
        self._respawn = []
        self._color = team_color
        self._flag = Flag(self)

    def add_player(self, player):
        assert player.team is None
        self._players.append(player)
        player.team = self

    def add_respawn(self, point):
        self._respawn.append(point)

    @property
    def respawns(self):
        return self._respawn

    @property
    def flag(self):
        return self._flag

    @property
    def players(self):
        return self._players

    @property
    def id(self):
        return self._id

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

    def render(self, img):
        if not self._flag.is_held:
            self._flag.render(img)
        # else:
        c = minigrid.COLORS[self.color]
        minigrid.fill_coords(img, minigrid.point_in_rect(0.20, 0.60, 0.78, 0.99), c)

    def can_pickup(self):
        return True

    def encode(self):
        return minigrid.OBJECT_TO_IDX[self.type], self.id, self.flag.is_held


class Player(Agent):

    def __init__(self, agent_id, state, health, respawn_after):
        super().__init__(agent_id, state)
        self._base_health = health
        self._health = health
        self._active = True
        self._team = None
        self._holding = None
        self._time_respawn = 0
        self._respawn_after = respawn_after

    @property
    def team(self):
        return self._team

    @team.setter
    def team(self, team):
        self._team = team

    def hit(self):
        self._health = max(0, self._health - 1)
        if self._health == 0:
            self._time_respawn = self._respawn_after

    def tick(self):
        self._time_respawn = max(0, self._time_respawn - 1)

    def hold(self, flag):
        self._holding = flag

    def tag(self, player):
        can_tag = self._team.id != player.team.id
        if can_tag:
            player.hit()
        return can_tag

    @property
    def holding(self):
        return self._holding

    def drop(self):
        assert self._holding is not None
        self._holding.drop(self)
        self._holding = None

    @property
    def is_holding(self):
        return self._holding is not None

    @property
    def health(self):
        return self._health

    def respawn(self):
        self._health = self._base_health

    def render(self, img):
        tri_fn = rendering.point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        tri_fn = rendering.rotate_fn(
            tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        c = minigrid.COLORS[self.team.color]
        rendering.fill_coords(img, tri_fn, c)

        if self.is_holding:
            c = minigrid.COLORS[self._holding.team.color]
            # Vertical quad
            post = minigrid.point_in_rect(0.20, 0.25, 0.20, 0.45)
            post = minigrid.rotate_fn(post, cx=0.5, cy=0.5, theta=0.5 * np.pi * (self.dir))
            minigrid.fill_coords(img, post, c)

            flag = minigrid.point_in_triangle(
                (0.20, 0.20),
                (0.20, 0.10),
                (0.40, 0.15),
            )
            flag = minigrid.rotate_fn(flag, cx=0.5, cy=0.5, theta=0.5 * np.pi * (self.dir))
            minigrid.fill_coords(img, flag, c)

    def encode(self):
        return minigrid.OBJECT_TO_IDX[self.type], self.agent_id, self.is_holding, self.dir

    @property
    def active(self):
        return self._time_respawn == 0


class RespawnPool:

    def __init__(self):
        self.players = []

    def add_player(self, player):
        self.players.append(player)

    def tick(self):
        respawned = []

        for player in self.players:
            player.tick()
            if player.active:
                respawned.append(player)
                self.players.remove(player)

        return respawned