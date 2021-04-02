import math
from gym_minigrid import minigrid, rendering
from multigym import multigrid


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
    def __init__(self, agent_id, state, team_id, team_color):
        super().__init__(agent_id, state)
        self.team_color = team_color
        self.team_id = team_id

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

    def __init__(self, team_id, color):
        super(Flag, self).__init__(type='goal', color=color)
        self.team_id = team_id

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


class Ray(minigrid.Wall):
    def __init__(self, color):
        super().__init__(color=color)

    def can_overlap(self):
        return True
