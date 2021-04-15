import os
import gym
import matplotlib.pyplot as plt

import multigym.ctf

SAMPLES = 50


def main():
    env = gym.make('MultiGrid-CTF-Classic-v0')
    env.reset()
    os.makedirs('samples', exist_ok=True)

    for i in range(SAMPLES):
        plt.imsave(f'samples/env_{i}.png', env.render('rgb', highlight=False))
        env.reset()


if __name__ == '__main__':
    main()
