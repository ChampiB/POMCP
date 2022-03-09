from srcs.environments.MazeEnv import MazeEnv
from srcs.agent.POMCP import POMCP
import time
import math

if __name__ == "__main__":

    # Create the environment
    MAZE_FILE_NAME = "5.maze"
    env = MazeEnv("./mazes/" + MAZE_FILE_NAME)
    LOCAL_MINIMA = env.get_local_minima(MAZE_FILE_NAME)
    S = env.states_list()
    A = env.actions_list()
    O = env.observations_list()

    # Hyper-parameters
    NB_SIMULATIONS = 100
    NB_ACTION_PERCEPTION_CYCLES = 20
    GAMMA = 0.9
    TIMEOUT = 500
    NO_PARTICLES = 100
    EXP_CONST = 5

    # Performance tracking variables
    exec_times = []
    perf = [0 for i in range(0, len(LOCAL_MINIMA) + 2)]

    # Run the simulations
    for j in range(0, NB_SIMULATIONS):

        # Create the agent
        agent = POMCP(S, A, O, env, c=EXP_CONST, gamma=GAMMA, timeout=TIMEOUT, no_particles=NO_PARTICLES)

        # Action-perception cycles
        start = time.time()
        # env.render()  # TODO
        for t in range(0, NB_ACTION_PERCEPTION_CYCLES):
            action = agent.search()
            obs, _, done = env.step(action)
            # env.render()  # TODO
            if done:
                break
            agent.update_belief(action, obs)
            agent.tree.prune_after_action(action, obs)

        # Track execution time and performance
        exec_times.append(time.time() - start)
        perf = env.track(perf, LOCAL_MINIMA)

        # Reset the environment to its initial state
        env.reset()

    # Display hyperparameters setting
    print("MAZE_FILE_NAME={}".format(MAZE_FILE_NAME))
    print("NB_SIMULATIONS={}".format(NB_SIMULATIONS))
    print("NB_ACTION_PERCEPTION_CYCLES={}".format(NB_ACTION_PERCEPTION_CYCLES))
    print("GAMMA={}".format(GAMMA))
    print("TIMEOUT={}".format(TIMEOUT))
    print("NO_PARTICLES={}".format(NO_PARTICLES))
    print("EXP_CONST={}".format(EXP_CONST))
    print()

    # Display execution time and performance of the POMCP agent
    avg = sum(exec_times) / len(exec_times)
    var = sum((x - avg)**2 for x in exec_times) / (len(exec_times) - 1)
    print("Time: {} +/- {}\n".format(avg, math.sqrt(var)))

    total = sum(perf)
    print("P(global): {}".format(perf[len(perf) - 1] / total))
    for i in range(0, len(LOCAL_MINIMA)):
        print("P(local {}): {}".format(i + 1, perf[i + 1] / total))
    print("P(other): {}".format(perf[0] / total))
