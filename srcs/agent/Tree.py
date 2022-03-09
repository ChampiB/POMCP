from srcs.agent.auxilliary import NodeAttr


#
# Class representing the search tree.
#
class Tree:

    def __init__(self):
        """
        Construct the search tree.
        """

        # Current node index
        self.current_id = -1

        # Create the root node of the tree
        self.nodes = {
            self.current_id: ['isRoot', {}, 0, 0, []]
        }

    def expand_tree_from(self, parent, index, is_action=False):
        """
        Add a new node to the tree
        :param parent: the parent of the new node
        :param index: the index of the observation or action that the new node represents
        :param is_action: whether the index represents an action or an observation
        :return: nothing
        """
        self.current_id += 1
        if is_action:
            self.nodes[self.current_id] = [parent, {}, 0, 0, -1]
            self.nodes[parent][NodeAttr.CHILDREN][index] = self.current_id
        else:
            self.nodes[self.current_id] = [parent, {}, 0, 0, []]
            self.nodes[parent][NodeAttr.CHILDREN][index] = self.current_id

    def is_leaf_node(self, n):
        """
        Check whether the input node is a leaf node
        :param n: the node index
        :return: true if the input node a leaf node, false otherwise.
        """
        return len(self.nodes[n][NodeAttr.CHILDREN]) == 0

    def get_obs_node(self, n, obs):
        """
        Returns the child of the node 'n' corresponding to the observation 'obs'
        :param n: the whose child must be returned
        :param obs: the observation
        :return: the node's child corresponding to the observation
        """
        # Get the children of the node 'n'
        children = self.nodes[n][NodeAttr.CHILDREN]

        # Create the child node, if it does not already exist
        if obs not in list(children.keys()):
            self.expand_tree_from(n, obs)

        # Return the child corresponding to the observation 'obs'
        return children[obs]

    def prune(self, node):
        """
        Remove a node from the tree as well as all its children
        :param node: the node that should be removed.
        :return: nothing.
        """
        children = self.nodes[node][NodeAttr.CHILDREN]
        del self.nodes[node]
        for _, child in children.items():
            self.prune(child)

    def set_new_root(self, new_root):
        """
        Change the root of the tree
        :param new_root: the new root node.
        :return: nothing.
        """

        # Replace the old root node by the new one.
        self.nodes[-1] = self.nodes[new_root].copy()
        del self.nodes[new_root]
        self.nodes[-1][NodeAttr.PARENT] = 'isRoot'

        # Update the index of the children's parent
        for _, child in self.nodes[-1][NodeAttr.CHILDREN].items():
            self.nodes[child][NodeAttr.PARENT] = -1

    def prune_after_action(self, action, obs):
        """
        Keep only the part of the tree that is relevant to the action taken and observation made.
        The irrelevant nodes are prunned
        :param action: the action taken
        :param obs: the observation made
        :return: nothing.
        """

        # Get the root's child corresponding the selected action.
        action_node = self.nodes[-1][NodeAttr.CHILDREN][action]

        # Get the action_node's child corresponding the observation made.
        new_root = self.get_obs_node(action_node, obs)

        # Prune the unnesecary nodes.
        del self.nodes[action_node][NodeAttr.CHILDREN][obs]
        self.prune(-1)

        # Update the tree's root.
        self.set_new_root(new_root)
