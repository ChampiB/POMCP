from numpy.random import choice
from srcs.agent.Tree import Tree
from srcs.agent.auxilliary import ucb
from srcs.agent.auxilliary import NodeAttr as NodeAttr


#
# Class implementing the Partially Observable Monte Carlo Planning algorithm.
#
class POMCP:

    def __init__(self, states, actions, obs, generator, gamma=0.95, c=1, threshold=0.005, timeout=10000, no_particles=1200):
        """
        Construct the POMCP agent
        :param states: the set of all possible states
        :param actions: the set of all possible actions
        :param obs: the set of all possible observations
        :param generator: the black box generator, i.e. environment
        :param gamma: the discount factor
        :param c: the exploration constant
        :param threshold: the threshold below which discount is too small
        :param timeout: the number of runs from node
        :param no_particles: the number of particle in the filter
        """

        # Check that the parameters are valid.
        if gamma >= 1:
            raise ValueError("gamma should be less than 1.")

        # Initialize the attributes.
        self.gamma = gamma
        self.generator = generator
        self.e = threshold
        self.c = c
        self.timeout = timeout
        self.no_particles = no_particles
        self.tree = Tree()
        self.states = states
        self.actions = actions
        self.observations = obs

    def best_child(self, node, use_ucb=True):
        """
        Finds and returns the child with the highest value
        If the parameter use_UCB=True, then the child's value is the UCB criterion
        If the parameter use_UCB=False, then the child's value is the action's value
        :param node: the node whose best child should be returned
        :param use_ucb: whether the UCB or the action's value should be used
        :return: the best child
        """

        # Check that the beliefs are valid
        if self.tree.nodes[node][NodeAttr.BELIEFS] == -1:
            print("Invalid beliefs over state in POMCP.best_child.")
            return None, None

        # Find and return the best action and associated child
        children = self.tree.nodes[node][NodeAttr.CHILDREN]
        max_action = max(
            children,
            key=lambda action: self.compute_node_value(children[action], use_ucb)
        )
        return max_action, children[max_action]

    def compute_node_value(self, node, use_ucb=True):
        """
        Compute the value of the input node
        :param node: the node whose value must be computed
        :param use_ucb: whether the UCB or the action's value should be used
        :return: the node's value
        """
        if use_ucb:
            parent = self.tree.nodes[node][NodeAttr.PARENT]
            return ucb(
                self.tree.nodes[parent][NodeAttr.VISITS],
                self.tree.nodes[node][NodeAttr.VISITS],
                self.tree.nodes[node][NodeAttr.VALUE],
                self.c
            )
        return self.tree.nodes[node][NodeAttr.VALUE]

    def search(self):
        """
        Perform the POMCP search.
        :return: the best action according to the search.
        """

        # Get belief state of the root node
        bh = self.root()[NodeAttr.BELIEFS].copy()

        # Repeat simulations until timeout
        for _ in range(self.timeout):
            self.simulate(self.sample_belief_state(bh), -1, 0)

        # Get best action
        return self.best_child(-1, use_ucb=False)[0]

    def rollout(self, s, depth):
        """
        Perform a roolout run from state 's'
        :param s: the state from which the rollout must be performed
        :param depth: the current depth of the rollout
        :return: the value of the rollout run
        """

        # Check significance of update
        if (self.gamma ** depth < self.e or self.gamma == 0) and depth != 0:
            return 0

        # Pick random action
        action = choice(self.actions)
        sample_state, _, r = self.generator(s, action)

        # Compute action's value
        return r + self.gamma * self.rollout(sample_state, depth + 1)

    def simulate(self, s, h, depth):
        """
        Perform one iteration of MCTS, i.e. one simulation through the tree
        :param s: the state from which the planning starts
        :param h: the tree's node from which the simulation starts
        :param depth: the current depth of the simulation
        :return: the value of the simulation
        """

        # Check significance of update
        if (self.gamma ** depth < self.e or self.gamma == 0) and depth != 0:
            return 0

        # If the current node is a leaf of the tree
        if self.tree.is_leaf_node(h):

            # Exapand all possible actions
            for action in self.actions:
                self.tree.expand_tree_from(h, action, is_action=True)

            # Perform a rollout to evaluate the value of the current state
            return self.rollout(s, depth)

        # Get best action and associated tree's node
        best_action, best_node = self.best_child(h)
        # Generate next state, observation and reward
        next_s, next_obs, reward = self.generator(s, best_action)
        # Get tree's node associated to the next observation
        next_node = self.get_observation_node(best_node, next_obs)
        # Estimate node Value
        cum_reward = reward + self.gamma * self.simulate(next_s, next_node, depth + 1)
        # Add current state to belief state
        self.tree.nodes[h][NodeAttr.BELIEFS].append(s)
        if len(self.tree.nodes[h][NodeAttr.BELIEFS]) > self.no_particles:
            self.tree.nodes[h][NodeAttr.BELIEFS] = self.tree.nodes[h][NodeAttr.BELIEFS][1:]
        # Back-propagate value and number of visits
        self.tree.nodes[h][NodeAttr.VISITS] += 1
        self.tree.nodes[best_node][NodeAttr.VISITS] += 1
        self.tree.nodes[best_node][NodeAttr.VALUE] += \
            (cum_reward - self.tree.nodes[best_node][NodeAttr.VALUE]) / self.tree.nodes[best_node][NodeAttr.VISITS]
        return cum_reward

    # Check if a given observation node has been visited
    def get_observation_node(self, h, obs):
        """
        Get the child of 'h' corresponding to the observation 'obs'.
        The node is created if it does not exist yet
        :param h: the node whose child should be returned
        :param obs: the observation specifying which child to return
        :return: the child of 'h' corresponding to the observation 'obs'.
        """

        # Add the observation to the tree if not already in it
        children = self.tree.nodes[h][NodeAttr.CHILDREN]
        if obs not in list(children.keys()):
            self.tree.expand_tree_from(h, obs)

        # Get the index of the node corresponding to the input observation
        return children[obs]

    def sample_belief_state(self, beliefs):
        """
        Sample the belief state, or the prior over state if the
        belief state does not contain any particle
        :param beliefs: the belief state
        :return: the sampled state
        """
        return choice(beliefs) if len(beliefs) != 0 else self.generator.current_state()  # TODO choice(self.states)

    def sample_posterior(self, Bh, a, obs):
        """
        Samples from posterior after taking action 'a' and receiving observation 'obs'
        :param Bh: the belief state
        :param a: the action taken by the agent
        :param obs: the observation received by the agent
        :return: the sample from the posterior
        """

        # Sample from belief state
        s = self.sample_belief_state(Bh)

        # Simulate action in the environment, i.e. sample transition distribution
        s_next, o_next, _ = self.generator(s, a)

        # If the observation matches, then return the state
        if o_next == obs:
            return s_next

        # Otherwise, sample a new state
        return self.sample_posterior(Bh, a, obs)

    def update_belief(self, action, observation):
        """
        Updates belief by sampling posterior
        :param action: the action taken by the agent
        :param observation: the observation received by the agent
        :return: nothing
        """

        # Retreive current belief state
        root = self.root()
        prior = root[NodeAttr.BELIEFS].copy()

        # Compute new belief state
        root[NodeAttr.BELIEFS] = []
        for _ in range(self.no_particles):
            root[NodeAttr.BELIEFS].append(self.sample_posterior(prior, action, observation))

    def root(self):
        """
        Getter
        :return: the tree's root.
        """
        return self.tree.nodes[-1]
