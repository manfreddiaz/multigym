from multigym.ctf.arena import ArenaGenerator


def test_deepmind_arena():
    arena = ArenaGenerator()
    print(arena)
    arena.regenerate()
    print(arena)


if __name__ == '__main__':
    test_deepmind_arena()