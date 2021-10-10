from gym_gridworld import GridWorldEnv

import time
import numpy as np

MAX_STEPS = 500

if __name__ == "__main__":
    env = GridWorldEnv()

    start = time.time()

    env.reset()

    i = 0
    env.render()
    points = 0
    while i < MAX_STEPS:
        rand = np.random.choice(env.action_space)
        _, reward, done, _ = env.step(rand)
        i += 1

        print("time step : " +str(i))
        env.render()

        if done:
            points = points+reward
            env.reset()

    end = time.time()

    time_per_step = (end - start) / MAX_STEPS
    print(f"Time per step: {(time_per_step):.2} s")
    print("points : " + str(points))

    env.close()