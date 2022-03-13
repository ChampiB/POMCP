from srcs.environments.LakeEnv import LakeEnv
from srcs.agent.POMCP import POMCP
import time
import math

if __name__ == "__main__":

    # Create the environment
    LAKE_FILE_NAME = "5.lake"
    env = LakeEnv("./lakes/" + LAKE_FILE_NAME)
    S = env.states_list()
    A = env.actions_list()
    O = env.observations_list()

    # Hyper-parameters
    NB_SIMULATIONS = 100
    NB_ACTION_PERCEPTION_CYCLES = 30
    GAMMA = 0.9
    TIMEOUT = 1000
    NO_PARTICLES = 100
    EXP_CONST = 3

    # Performance tracking variables
    exec_times = []
    perf = [0, 0]

    # Run the simulations
    for j in range(0, NB_SIMULATIONS):

        # Create the agent
        agent = POMCP(S, A, O, env, c=EXP_CONST, gamma=GAMMA, timeout=TIMEOUT, no_particles=NO_PARTICLES)

        # Action-perception cycles
        start = time.time()
        for t in range(0, NB_ACTION_PERCEPTION_CYCLES):
            action = agent.search()
            obs, _, done = env.step(action)
            if done:
                break
            agent.update_belief(action, obs)
            agent.tree.prune_after_action(action, obs)

        # Track execution time and performance
        exec_times.append(time.time() - start)
        perf = env.track(perf, [])

        # Reset the environment to its initial state
        env.reset()

    # Display hyperparameters setting
    print("LAKE_FILE_NAME={}".format(LAKE_FILE_NAME))
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
    print("P(global): {}".format(perf[1] / total))
    print("P(other): {}\n".format(perf[0] / total))

    print("Number of holes: {}".format(env.nb_holes))
