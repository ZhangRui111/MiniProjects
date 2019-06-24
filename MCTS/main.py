"""
A Monte Carlo Tree Search implementation for maze based on haroldsultan
at https://github.com/haroldsultan/MCTS/issues.
"""

import random
import math
import numpy as np
import hashlib

from env.maze import Maze

# MCTS scalar: exploitation_value + scalar \times exploration_value
SCALAR = 1 / math.sqrt(2.0)
MOVES = [0, 1, 2, 3]  # up, down, left, right.
NUM_MOVES = len(MOVES)


class CState(object):
    def __init__(self, state, reward=0, terminal=False, moves=None):
        self.state = state
        self.reward = reward
        self.terminal = terminal
        if moves is None:
            self.moves = []
        else:
            self.moves = moves

    def next_state(self):
        # next_m, next_v, r, terminal, clear_traj = env_random_step()
        next_m, next_v, r, terminal = env_random_step()
        if self.terminal:
            next_s = CState(next_v, r, terminal, [next_m])
        else:
            next_s = CState(next_v, r, terminal, self.moves + [next_m])
        return next_s

    def next_reward(self):
        """ Return the reward of next_state. """
        next_s = self.next_state()
        sum_reward = next_s.reward
        while next_s.terminal is False:
            next_s = next_s.next_state()
            sum_reward += next_s.reward
        env.render(0.01)
        env.reset(init_state)
        self.terminal = False
        # raise Exception("State is terminal!")
        return sum_reward

    def __hash__(self):
        # hashlib.md5().hexdigest() => get the hexadecimal digest of the strings fed to it.
        return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        s = "Value: {0} | Reward: {1} | Terminal: {2} | Moves: {3}".\
            format(self.state.ravel(), self.reward, self.terminal, self.moves)
        return s


class CNode(object):
    def __init__(self, cstate, parent=None):
        self.visits = 0  # how many times this note is visited.
        self.cstate = cstate
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        child = CNode(child_state, parent=self)
        self.children.append(child)

    def update(self, reward):
        self.cstate.reward += reward
        self.visits += 1

    def fully_explored(self):
        if len(self.children) == NUM_MOVES:
            return True
        return False

    def __repr__(self):
        # __repr__() is What to show when we print a sample of this class.
        s = "Node: children: {0} | visits: {1} | reward: {2}"\
            .format(len(self.children), self.visits, self.cstate.reward)
        return s


def explore(node):
    """ Explore a new note with new state/action and return this note."""
    tried_children = [c.cstate for c in node.children]
    new_state = node.cstate.next_state()
    while new_state in tried_children:
        new_state = node.cstate.next_state()
    node.add_child(new_state)
    return node.children[-1]


def best_child(node, scalar):
    """ Return a child node with the best score. """
    best_score = -1000  # A very low best_score to assure the first score is larger than this value.
    best_children = []
    for c in node.children:
        exploit_v = c.cstate.reward / c.visits
        explore_v = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
        score = exploit_v + scalar * explore_v  # the most vanilla MCTS formula.
        if score == best_score:
            best_children.append(c)
        if score > best_score:
            best_children = [c]
            best_score = score
    if len(best_children) == 0:
        raise Exception("OOPS: no best child found, probably fatal.")
    return random.choice(best_children)


def tree_policy(node):
    """
    A policy to 'exploitation' with a certain possibility
        in a game where there are many options.
    You may never/not want to fully explore first.
    :param node:
    :return:
    """
    while node.cstate.terminal is False:
        if len(node.children) == 0:
            # Create a child node when there is no child node.
            return explore(node)
        elif random.uniform(0, 1) < 0.5:
            # exploit the best child node with a certain possibility.
            return best_child(node, SCALAR)
        else:
            # explore a new node if not fully explored;
            # otherwise, exploit the best child node.
            if node.fully_explored() is False:
                return explore(node)
            else:
                return best_child(node, SCALAR)


def backup(node, reward):
    """
    Update the node.visits and node.reward from the child
    node to the parent node iteratively.
    """
    while node is not None:
        node.visits += 1
        node.cstate.reward += reward
        node = node.parent


def env_random_step():
    """ Adopt random action in the env. """
    # Signal so that we can clear useless trajectory once terminal in the hell.
    # clear_trajectory = False
    env.render(0.01)
    a = random.choice(MOVES)
    s_, r, terminal, info = env.step(a)
    # if terminal:
    #     env.render(0.01)
    #     env.reset()
    #     terminal = False
    #     # clear_trajectory = True
    # return a, s_, r, terminal, clear_trajectory
    return a, s_, r, terminal


def uct_search(num_sims, root):
    """
    UCT is the algorithm used in the vast majority of
        current MCTS implementations.
    UCT = MCTS + UCB
    """
    # Sampling to get accurate rewards for every nodes.
    for iter in range(int(num_sims)):
        selected_node = tree_policy(root)
        reward = selected_node.cstate.next_reward()
        backup(selected_node, reward)
    # fully exploit to get the best child node.
    print("iteration over!")
    return best_child(root, scalar=0)


def update_maze_terminal(state):
    """ Determine whether we have solve the env. """
    terminal_state = np.asarray([0, 0, -1, 0, 0, 0, -1, 0, 1])
    # terminal_state = np.asarray([0, 1, -1, 0, 0, 0, -1, 0, 2])
    # term_a = list(map(int, state.ravel()))
    # term_b = terminal_state.ravel().tolist()
    if list(map(int, state.ravel())) == terminal_state.ravel().tolist():
        return True
    else:
        return False


def run_maze():
    num_sims = 100  # Number of simulations to run.
    global init_state
    init_state = [[0, 0]]

    # MCTS sampling for more accurate evaluation.
    observation = env.reset(init_state)
    current_node = CNode(CState(observation))
    maze_terminal = False

    while not maze_terminal:
        # player's position of the current root state.
        row_column = np.where(current_node.cstate.state == 1)
        init_state = [[int(row_column[0]), int(row_column[1])]]
        print("init_state: {0}".format(init_state))

        # # Make sure the env.s equals to current_node.cstate.state
        # if len(current_node.cstate.moves) != 0:
        #     for a in current_node.cstate.moves:
        #         env.step(a)

        print("Current node: \n{0}".format(current_node.cstate.state))
        current_node = uct_search(num_sims, current_node)
        print("Number of children: {0}".format(len(current_node.parent.children)))
        for i, c in enumerate(current_node.children):
            print(i, c)
        print("Best child: {0}".format(current_node.cstate))
        print("Best child: \n{0}".format(current_node.cstate.state))
        print("-----------------------------------------")
        maze_terminal = update_maze_terminal(current_node.cstate.state)

    # MCTS sampling over & find the best trajectory.
    print("MCTS sampling over")
    init_state = [[0, 0]]
    env.reset(init_state)
    while current_node.parent is not None:
        current_node = current_node.parent

    while len(current_node.children) != 0:
        print("Current node: \n{0}".format(current_node.cstate.state))
        current_node = best_child(current_node, scalar=0)
        # print("Number of children: {0}".format(len(current_node.children)))
        # for i, c in enumerate(current_node.children):
        #     print(i, c)
        print("Best child: {0}".format(current_node.cstate))
        print("Best child: \n{0}".format(current_node.cstate.state))
        print("-----------------------------------------")

    # end of game
    print("Game over")
    env.destroy()


def main():
    global env
    env = Maze('./env/maps/map1.json', full_observation=True)
    env.after(100, run_maze)  # Call function update() once after given time/ms.
    env.mainloop()  # mainloop() to run the application.


if __name__ == '__main__':
    main()
