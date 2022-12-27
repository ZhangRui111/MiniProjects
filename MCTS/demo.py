"""
A quick Monte Carlo Tree Search implementation by haroldsultan at
    https://github.com/haroldsultan/MCTS.

For more details on MCTS:
See https://www.cs.swarthmore.edu/~mitchell/classes/cs63/f20/reading/mcts.html

The State is just a game where you have MAX_NUM_TURNS and
    at turn i you can make a choice from [-2,2,3,-3]*i
    and this to an accumulated value.
The goal is for the accumulated value to be as close to 0 as possible.
The game is not very interesting but it allows one to study MCTS which is.
Some features of the example by design are that moves do not commute and
    early mistakes are more costly.
In particular there are two models of best child that one can use.
"""

import random
import math
import hashlib
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')

# exploitation_value + scalar \times exploration_value
# Larger scalar will increase exploration, smaller will increase exploitation.
EXPLO_SCALAR = 1 / math.sqrt(2.0)
MAX_NUM_TURNS = 10
GOAL = 0
MOVES = [2, -2, 3, -3]
NUM_MOVES = len(MOVES)
MAX_VALUE = (5.0 * (MAX_NUM_TURNS - 1) * MAX_NUM_TURNS) / 2  # 225.0


class State(object):
    def __init__(self, value=0, moves=[], turn=MAX_NUM_TURNS):
        # The accumulated value to be as close to 0 as possible.
        self.value = value  # the cumulated value, corresponding to the GOAL
        self.turn = turn  # self.turn decreases from its maximum value
        self.moves = moves  # the trajectory since the root node

    def next_state(self):
        # make a choice from [-2, 2, 3, -3] * self.turn
        nextmove = random.choice([x * self.turn for x in MOVES])
        next = State(self.value + nextmove, self.moves + [nextmove],
                     self.turn - 1)
        return next

    def terminal(self):
        if self.turn == 0:
            return True
        return False

    def reward(self):
        r = 1.0 - (abs(self.value - GOAL) / MAX_VALUE)
        return r

    def __hash__(self):
        # hashlib.md5().hexdigest() =>
        #     get the hexadecimal digest of the strings fed to it.
        return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        s = "Value: {:d} -- Moves: {}".format(self.value, self.moves)
        return s


class Node(object):
    def __init__(self, state, parent=None):
        self.visits = 1  # how many times this note is visited.
        self.reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        child = Node(child_state, parent=self)
        self.children.append(child)

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_explored(self):
        if len(self.children) == NUM_MOVES:
            return True
        return False

    def __repr__(self):
        s = "Node; children: {:d}; visits: {:d}; reward: {:f}".format(
            len(self.children), self.visits, self.reward)
        return s


def explore(node):
    """ Explore a new note with new state/action and return this note."""
    tried_children = [c.state for c in node.children]
    new_state = node.state.next_state()
    while new_state in tried_children:
        new_state = node.state.next_state()
    node.add_child(new_state)
    return node.children[-1]


def best_child(node, scalar):
    """ Return a child node with the best score. """
    bestscore = 0.0
    bestchildren = []
    for c in node.children:
        exploit_v = c.reward / c.visits
        explore_v = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
        score = exploit_v + scalar * explore_v  # the most vanilla MCTS formula.
        if score == bestscore:
            bestchildren.append(c)
        if score > bestscore:
            bestchildren = [c]
            bestscore = score
    if len(bestchildren) == 0:
        logger.warning("OOPS: no best child found, probably fatal.")
    return random.choice(bestchildren)


def tree_policy(node):
    """
    A policy to 'exploitation' with a certain possibility
        in a game where there are many options.
    You may never/not want to fully explore first.
    :param node:
    :return:
    """
    while node.state.terminal() is False:
        if len(node.children) == 0:
            # Create a child node when there is no child node.
            return explore(node)
        elif random.uniform(0, 1) < 0.5:
            # exploit the best child node with a certain possibility.
            node = best_child(node, EXPLO_SCALAR)
        else:
            # explore a new node if not fully explored;
            # otherwise, exploit the best child node.
            if node.fully_explored() is False:
                return explore(node)
            else:
                node = best_child(node, EXPLO_SCALAR)
    return node


def simulate(state):
    """ Return the reward of next_state. """
    while state.terminal() is False:
        state = state.next_state()
    return state.reward()


def backpropagation(node, reward):
    """
    Update the node.visits and node.reward from the child
    node to the parent node iteratively.
    """
    while node is not None:
        node.update(reward)
        node = node.parent


def uct_alg(num_sims, root):
    """
    UCT is the algorithm used in the vast majority of
        current MCTS implementations.
    UCT = MCTS + UCB
    """
    # Sampling to get accurate rewards for every nodes.
    for iter in range(int(num_sims)):
        if iter % 10000 == 9999:
            logger.info("simulation: {:d}".format(iter))
            logger.info(root)
        selected_node = tree_policy(root)
        reward = simulate(selected_node.state)
        backpropagation(selected_node, reward)
    # fully exploit (scalar=0) to get the best child node.
    return best_child(root, scalar=0)


if __name__ == "__main__":
    num_sims = 5000  # Number of simulations to run.
    num_levels = 10  # How many levels in a tree: <= MAX_NUM_TURNS

    current_node = Node(State())
    for lev in range(num_levels):
        # num_sims / (l + 1) =>
        #     the deeper level has less sampling to reduce computation
        # Because deeper the tree is, more trajectories the tree has.
        current_node = uct_alg(num_sims / (lev + 1), current_node)
        print("level {:d}".format(lev))
        print("Num Children: {:d}".format(len(current_node.children)))
        for i, c in enumerate(current_node.children):
            print(i, c)
        print("Best Child: {}\n--------------------------------".format(
            current_node.state))
