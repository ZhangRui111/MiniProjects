"""
Modified from https://gist.github.com/andrewliao11/d52125b52f76a4af73433e1cf8405a8f
For pseudo-code of CEM, please refer to wikipedia
https://en.wikipedia.org/wiki/Cross-entropy_method -- Continuous optimization—example
"""
import gym
import matplotlib.pyplot as plt
import numpy as np


def noisy_evaluation(env, W, render=False):
    """
    Uses parameter vector W to choose policy for 1 episode,
    and returns reward from that episode.
    """
    reward_sum = 0
    state = env.reset()
    t = 0
    while True:
        t += 1
        # use parameters/state to choose action
        action = int(np.dot(W, state) > 0)
        state, reward, done, info = env.step(action)
        reward_sum += reward
        if render and t % 3 == 0:
            env.render()
        if done or t > 205:
            # print("finished episode, got reward:{}".format(reward_sum))
            break

    return reward_sum


def init_params(mu, sigma, n):
    """
    Take vector of mus, vector of sigmas, create an initialized matrix.
    """
    l = mu.shape[0]
    w_matrix = np.zeros((n, l))
    for i in range(l):
        w_matrix[:, i] = np.random.normal(loc=mu[i], scale=abs(sigma[i]) + 1e-17, size=(n,))
    return w_matrix


def get_constant_noise(step):
    # return np.max(5 - step / 10., 0)
    return 0  # By exp., it seems no-noisy has higher rewards.


def main():
    env = gym.make('CartPole-v0')
    env.render()
    # Vector of means(mu) and Standard dev(sigma) for each parameter
    mu = np.random.uniform(size=env.observation_space.shape)
    sigma = np.random.uniform(low=0.001, size=env.observation_space.shape)

    running_reward = 0
    n_samples = 40
    n_top = 8
    n_iter = 50
    render = False

    i = 0
    while i < n_iter:
        # Initialize an array of parameter vectors.
        w_vector_array = init_params(mu, sigma, n_samples)
        reward_sums = np.zeros(n_samples)
        for k in range(n_samples):
            # Sample rewards based on policy parameters in row k of w_vector_array.
            reward_sums[k] = noisy_evaluation(env, w_vector_array[k, :], render)
        env.render()

        # Sort params/vectors based on total reward of an episode using that policy.
        rankings = np.argsort(reward_sums)
        # Pick n_top vectors with highest reward.
        top_vectors = w_vector_array[rankings, :]
        top_vectors = top_vectors[-n_top:, :]
        print("top vectors shpae:{}".format(top_vectors.shape))

        # Fit new gaussian from which to sample policy.
        for j in range(top_vectors.shape[1]):
            mu[j] = top_vectors[:, j].mean()
            sigma[j] = top_vectors[:, j].std() + get_constant_noise(j)
            # print(get_constant_noise(j), sigma[j])

        running_reward = 0.99 * running_reward + 0.01 * reward_sums.mean()
        print("#############################################################################")
        print("iteration:{0}, mean reward:{1}, running reward mean:{2}, reward range:{3} to {4}"
              .format(i, reward_sums.mean(), running_reward, reward_sums.min(), reward_sums.max()))
        i += 1


if __name__ == '__main__':
    main()
