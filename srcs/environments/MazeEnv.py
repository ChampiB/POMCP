import numpy as np
import torch
from PIL import Image
from srcs.environments.viewers.DefaultViewer import DefaultViewer


#
# This file contains the code of the Maze environment.
#
class MazeEnv:

    def __init__(self, maze_file):
        """
        Constructor.
        :param maze_file: the file containing the maze's description.
        """

        # Initialize fields
        self.exit_pos = [-1, -1]
        self.agent_pos = [-1, -1]
        self.last_r = 0  # Last reward
        self.scale = 10  # How big should each cell be represented (in pixels) ?
        nb_states = 0

        # Load maze from file
        file = open(maze_file, "r")
        maze = file.readlines()

        h, w = maze[0].split(" ")
        h = int(h)
        w = int(w)
        maze = maze[1:h+1]
        maze = [line.rstrip('\n') for line in maze]
        self.maze = torch.ones((h, w))

        for i in range(0, h):
            for j in range(0, w):
                if maze[i][j] == 'W':
                    self.maze[i][j] = 1
                elif maze[i][j] == '.':
                    nb_states += 1
                    self.maze[i][j] = 0
                elif maze[i][j] == 'E':
                    nb_states += 1
                    self.maze[i][j] = 0
                    self.exit_pos[0] = i
                    self.exit_pos[1] = j
                elif maze[i][j] == 'S':
                    nb_states += 1
                    self.maze[i][j] = 0
                    self.agent_pos[0] = i
                    self.agent_pos[1] = j
                else:
                    raise Exception("Invalid file format: '" + maze_file + "'")

        self.agent_initial_pos = [self.agent_pos[0], self.agent_pos[1]]
        self.reset()

        # Action, state, and observation spaces
        self.action_space = [0, 1, 2, 3]  # [up, down, right, left]
        self.spate_space = [i for i in range(0, nb_states)]
        self.observation_space = [i for i in range(0, h + w - 5)]

        # Graphical interface
        self.viewer = None

        # States indices
        self.states_idx = None  # Mapping from (x,y) positions to state indices
        self.load_states_indices()
        self.states_positions = {}  # Mapping from state indices to (x,y) positions
        self.load_states_positions()

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        :return: the first observation.
        """
        self.agent_pos[0] = self.agent_initial_pos[0]
        self.agent_pos[1] = self.agent_initial_pos[1]
        return self.current_frame()

    def __call__(self, s, a):
        """
        Simulate the execution of an action in the environment from a specific state.
        :param s: the state from which to perform the action.
        :param a: the action to perform, i.e. UP, DOWN, LEFT, or RIGHT.
        :return: next observation, reward, is the trial done?, information
        """
        next_s, obs, r, _ = self.simualte_action(s, a)
        return next_s, obs, r

    def step(self, action):
        """
        Execute one action within the environment.
        :param action: the action to perform, i.e. UP, DOWN, LEFT, or RIGHT.
        :return: next observation, reward, is the trial done?
        """
        state = self.states_idx[self.agent_pos[0]][self.agent_pos[1]].item()
        next_state, obs, self.last_r, done = self.simualte_action(state, action)
        x_pos, y_pos = self.states_positions[next_state]
        self.agent_pos[0] = y_pos
        self.agent_pos[1] = x_pos
        return obs, self.last_r, done

    def simualte_action(self, state, action):
        """
        Simulate the impact of taking an action in a specific state.
        :param state: the state from each the action will be simulated.
        :param action: the action taken.
        :return: the next state, observation and reward, as well as a boolean telling whether the simulation ended.
        """

        # Check validity of parameters
        if not isinstance(action, int):
            action = action.item()
        if action < 0 or action > 3:
            exit('Invalid action.')

        # Simulate the action
        actions_fn = [self.up, self.down, self.right, self.left]
        done, x_pos, y_pos = actions_fn[action](state)

        # Compute the reward
        s = self.maze.size()[0] + self.maze.size()[1]
        reward = (s - self.manhattan_distance((y_pos, x_pos), self.exit_pos)) / s

        # Return the state, observation, reward, and whether the simulation is done or not
        obs = self.manhattan_distance((y_pos, x_pos), self.exit_pos)
        return self.states_idx[y_pos][x_pos].item(), obs, reward, done

    @staticmethod
    def manhattan_distance(pos1, pos2):
        """
        Compute the Manhattan distance between the two input positions.
        :param pos1: first input position.
        :param pos2: second input position.
        :return: the Manhattan distance.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def render(self):
        """
        Display the current state of the environment as an image.
        :return: nothing.
        """
        if self.viewer is None:
            self.viewer = DefaultViewer("Maze", round(self.last_r, 5), self.current_frame(), resize_type=Image.NEAREST)
        else:
            self.viewer.update(round(self.last_r, 5), self.current_frame())

    def current_frame(self):
        """
        Return the current frame (i.e. the current observation).
        :return: the current observation.
        """
        image = 1 - self.maze
        image[self.agent_pos[0]][self.agent_pos[1]] = 0.5
        image = image.numpy().astype(np.float64)
        return np.repeat(np.expand_dims(image, axis=2), 3, 2) * 255

    def current_state(self):
        """
        Getter.
        :return: the current state of the environment.
        """
        return self.states_idx[self.agent_pos[0]][self.agent_pos[1]].item()

    def is_solved(self, x_pos, y_pos):
        """
        Answer the question: Has the agent at position (x_pos, y_pos) reached the exit?
        :param x_pos: the x position of the agent.
        :param y_pos: the y position of the agent.
        :return: true if the agent reached the exit, false otherwise.
        """
        return y_pos == self.exit_pos[0] and x_pos == self.exit_pos[1]

    def load_states_indices(self):
        """
        Load the indices of each state.
        :return: nothing (the indices are loaded into the class attribute).
        """
        state_id = 0
        maze_shape = self.maze.size()
        self.states_idx = torch.full(maze_shape, -1).to(torch.int32)
        for j in range(0, maze_shape[0]):
            for i in range(0, maze_shape[1]):
                if self.maze[j][i] == 0:
                    self.states_idx[j][i] = state_id
                    state_id += 1

    def load_states_positions(self):
        """
        Load the positions of each state indice.
        :return: nothing (the positions are loaded into the class attribute).
        """
        maze_shape = self.maze.size()
        for j in range(0, maze_shape[0]):
            for i in range(0, maze_shape[1]):
                if self.states_idx[j][i] != -1:
                    self.states_positions[self.states_idx[j][i].item()] = (i, j)

    #
    # Actions
    #
    def up(self, state):
        """
        Perform the action "going up" in the environment.
        :param state: the state of the environment from which the action should be taken.
        :return: true if the end of the trial has been reached, false otherwise.
        """
        x_pos, y_pos = self.states_positions[state]
        if y_pos - 1 >= 0 and self.maze[y_pos - 1][x_pos] == 0:
            y_pos -= 1
        return self.is_solved(x_pos, y_pos), x_pos, y_pos

    def down(self, state):
        """
        Perform the action "going down" in the environment.
        :param state: the state of the environment from which the action should be taken.
        :return: true if the end of the trial has been reached, false otherwise.
        """
        x_pos, y_pos = self.states_positions[state]
        if y_pos + 1 < self.maze.shape[0] and self.maze[y_pos + 1][x_pos] == 0:
            y_pos += 1
        return self.is_solved(x_pos, y_pos), x_pos, y_pos

    def left(self, state):
        """
        Perform the action "going left" in the environment.
        :param state: the state of the environment from which the action should be taken.
        :return: true if the end of the trial has been reached, false otherwise.
        """
        x_pos, y_pos = self.states_positions[state]
        if x_pos - 1 >= 0 and self.maze[y_pos][x_pos - 1] == 0:
            x_pos -= 1
        return self.is_solved(x_pos, y_pos), x_pos, y_pos

    def right(self, state):
        """
        Perform the action "going right" in the environment.
        :param state: the state of the environment from which the action should be taken.
        :return: true if the end of the trial has been reached, false otherwise.
        """
        x_pos, y_pos = self.states_positions[state]
        if x_pos + 1 < self.maze.shape[1] and self.maze[y_pos][x_pos + 1] == 0:
            x_pos += 1
        return self.is_solved(x_pos, y_pos), x_pos, y_pos

    #
    # Action, state and observation space
    #
    def states_list(self):
        """
        Getter.
        :return: the list of possible states.
        """
        return self.spate_space

    def actions_list(self):
        """
        Getter.
        :return: the list of possible actions.
        """
        return self.action_space
        
    def observations_list(self):
        """
        Getter.
        :return: the list of possible observations.
        """
        return self.observation_space

    #
    # Performance tracking
    #
    @staticmethod
    def get_local_minima(key):
        """
        This function returns the position of the local minimum of the maze
        whose name is passed as parameters.
        :param key: the maze's name
        :return: a list containing the local minimum positions
        """
        local_minima = {
            "1.maze": [(3, 4)],
            "5.maze": [(3, 3)],
            "9.maze": [(3, 5), (5, 3)]
        }
        return local_minima[key]

    def track(self, perf, local_pos):
        tolerance = 1
        local_min = -1

        for i in range(0, len(local_pos)):
            if self.manhattan_distance(self.agent_pos, local_pos[i]) <= tolerance:
                local_min = i
        if self.manhattan_distance(self.agent_pos, self.exit_pos) <= tolerance:
            perf[len(perf) - 1] += 1
        elif local_min != - 1:
            perf[local_min + 1] += 1
        else:
            perf[0] += 1
        return perf
