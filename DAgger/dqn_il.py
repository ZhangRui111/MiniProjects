import numpy as np
import tensorflow as tf

from env.maze import Maze
from DQN_il_brain import DeepQNetwork
from utils import plot_rate, one_hot_encoding_numpy, get_batch, write_to_file

# Stimulated expert library.
ACTION_UP = [[4, 2], [4, 3], [5, 0], [5, 2]]
ACTION_DOWN = [[0, 0], [0, 1], [0, 4], [1, 1], [1, 4], [2, 2], [2, 4], [3, 5], [4, 5]]
ACTION_LEFT = [[0, 2], [0, 5]]
ACTION_RIGHT = [[1, 0], [2, 1], [3, 2], [3, 3], [3, 4], [4, 0], [4, 1]]


def get_expert_action(obser):
    """ Get the expert action for every input observation. """
    pos = np.where(obser == 1)
    pos_ = [pos[0], pos[1]]
    if pos_ in ACTION_UP:
        return 0
    elif pos_ in ACTION_DOWN:
        return 1
    elif pos_ in ACTION_LEFT:
        return 2
    elif pos_ in ACTION_RIGHT:
        return 3
    else:
        raise Exception("invalid observation in get_expert_action()")


def init_model():
    n_features = env.height * env.width
    n_actions = 4
    render_time = 0

    obser_list = []
    action_list = []
    obsers_all = np.zeros((1, n_features))
    actions_all = np.zeros((1, 1))

    # # Collecting data.
    for j in range(200):
        s = env.reset()
        while True:
            env.render(render_time)
            a = get_expert_action(s)
            obser_list.append(s.ravel())
            action_list.append(a)
            s_, r, done, info = env.step(a)
            if done:
                env.render(render_time)
                break
            s = s_

    assert len(obser_list) == len(action_list)
    for obser, act in zip(obser_list, action_list):
        obsers_all = np.concatenate([obsers_all, obser[np.newaxis, :]], axis=0)
        actions_all = np.concatenate([actions_all, np.array([act])[np.newaxis, :]], axis=0)
    obsers_all = obsers_all[1:, :]
    actions_all = actions_all[1:, :].astype(int)
    actions_all = one_hot_encoding_numpy(actions_all.ravel().tolist(), 4)

    # # Training an initialized policy and store it.
    net_s = tf.placeholder(tf.float32, [None, n_features], name='net_s')  # observations
    net_a = tf.placeholder(tf.int32, [None, n_actions], name='net_a')  # expert actions
    with tf.variable_scope('net'):
        with tf.variable_scope('l1'):
            net_l1 = tf.contrib.layers.fully_connected(net_s, 32, activation_fn=tf.nn.relu)
            # net_l2 = tf.contrib.layers.fully_connected(net_l1, 128, activation_fn=tf.nn.relu)
        with tf.variable_scope('l2'):
            net_out = tf.contrib.layers.fully_connected(net_l1, n_actions, activation_fn=tf.nn.softmax)

    net_loss = tf.losses.softmax_cross_entropy(onehot_labels=net_a, logits=net_out)  # compute cost
    train_op = tf.train.AdamOptimizer(0.001).minimize(net_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(100):
            b_s, b_a = get_batch(obsers_all, actions_all, batch_size=32)
            _, loss_ = sess.run([train_op, net_loss], {net_s: b_s, net_a: b_a})

            if step % 5 == 0:
                output = sess.run(net_out, {net_s: obsers_all, net_a: actions_all})
                item_a = np.argmax(output, axis=1)
                item_b = np.argmax(actions_all, axis=1)
                n_accuracy = np.where(np.equal(np.argmax(output, axis=1), np.argmax(actions_all, axis=1)))[0]
                accuracy_ = n_accuracy.shape[0] / actions_all.shape[0]
                print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

        output = sess.run(net_out, {net_s: obsers_all, net_a: actions_all})
        n_accuracy = np.where(np.equal(np.argmax(output, axis=1), np.argmax(actions_all, axis=1)))[0]
        accuracy_ = n_accuracy.shape[0] / actions_all.shape[0]
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

        vars_list = [v for v in tf.global_variables()]
        write_to_file('./logs/dqn_il/model/vars_list.txt', vars_list, True)

        saver = tf.train.Saver()
        save_path = saver.save(sess, './logs/dqn_il/model/model_init.ckpt')
        print("Model saved in path: {}".format(save_path))

        return save_path


def run_maze():
    step = 0
    render_time = 0.1
    max_episodes = 1500
    episode_step_holder = []
    success_holder = []
    base_path = './logs/dqn_il/model/'

    for i_episode in range(max_episodes):
        episode_step = 0
        s = env.reset().ravel()

        while True:
            env.render(render_time)
            action = RL.choose_action(s)
            s_, reward, done, info = env.step(action)
            s_ = s_.ravel()
            # print('action:{0} | reward:{1} | done: {2}'.format(action, reward, done))
            RL.store_transition(s, action, reward, s_)

            if i_episode > 10:
                RL.learn(done)

            s = s_
            step += 1
            episode_step += 1

            if episode_step > 299:
                done = True

            if done:
                print('{0} -- {1} -- {2}'.format(i_episode, info, episode_step))
                if info == 'running':
                    episode_step = 300
                    success_holder.append(0)
                elif info == 'terminal':
                    success_holder.append(1)
                else:
                    raise Exception("Invalid info code.")
                env.render(render_time)
                episode_step_holder.append(episode_step)
                break

    # end of game
    print('game over')
    save_path = RL.saver.save(RL.sess, base_path + 'model_dqn.ckpt')
    print("Model saved in path: {}".format(save_path))
    RL.sess.close()
    env.destroy()

    # plot_cost(episode_step_holder, base_path + 'episode_steps.png')
    plot_rate(success_holder, base_path, index=15)


def enter_dqn_il():
    global env, RL
    env = Maze('./env/maps/map3.json', full_observation=True)
    save_path = init_model()
    tf.reset_default_graph()
    RL = DeepQNetwork(
        n_actions=4,
        n_features=env.height * env.width,
        restore_path=save_path,
        learning_rate=0.001,
        reward_decay=0.9,
        e_greedy=0.95,
        replace_target_iter=3000,
        batch_size=64,
        # e_greedy_init=0.9,
        e_greedy_init=0,
        # e_greedy_increment=None,
        e_greedy_increment=1e-3,
        output_graph=False,
    )
    env.after(100, run_maze)
    env.mainloop()


def main():
    global env, RL
    env = Maze('./env/maps/map3.json', full_observation=True)
    save_path = init_model()
    tf.reset_default_graph()
    RL = DeepQNetwork(
        n_actions=4,
        n_features=env.height*env.width,
        restore_path=save_path,
        learning_rate=0.001,
        reward_decay=0.9,
        e_greedy=0.95,
        replace_target_iter=3000,
        batch_size=64,
        # e_greedy_init=0.9,
        e_greedy_init=0,
        # e_greedy_increment=None,
        e_greedy_increment=1e-3,
        output_graph=False,
    )
    env.after(100, run_maze)
    env.mainloop()


if __name__ == "__main__":
    main()
