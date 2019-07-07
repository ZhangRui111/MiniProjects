import numpy as np
import tensorflow as tf

from env.maze import Maze
from utils import get_batch, one_hot_encoding_numpy, plot_rate


# Stimulated expert library.
ACTION_UP = [[4, 2], [4, 3], [5, 0], [5, 2]]
ACTION_DOWN = [[0, 0], [0, 1], [0, 4], [1, 1], [1, 4], [2, 2], [2, 4], [3, 5], [4, 5]]
ACTION_LEFT = [[0, 2], [0, 5]]
ACTION_RIGHT = [[1, 0], [2, 1], [3, 2], [3, 3], [3, 4], [4, 0], [4, 1]]

RUN_STEPS = 200
TRAIN_STEPS = 100
RENDER_TIME = 0
BASE_LOGS = './logs/dagger/model/'


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


def run_maze():
    # expert_counter = 0  # Record how many times we refer the expert action.
    n_features = env.height * env.width
    n_actions = 4
    restore_path = None
    dagger_itr = 5  # how many times we do dataset aggregation.
    episode_step_holder = []
    success_holder = []

    obser_list = []
    action_list = []
    obsers_all = np.zeros((1, n_features))
    actions_all = np.zeros((1, 1))

    # # Collecting data.
    for j in range(RUN_STEPS):
        s = env.reset()
        while True:
            env.render(RENDER_TIME)
            a = get_expert_action(s)
            obser_list.append(s.ravel())
            action_list.append(a)
            s_, r, done, info = env.step(a)
            if done:
                env.render(RENDER_TIME)
                break
            s = s_

    assert len(obser_list) == len(action_list)
    for obser, act in zip(obser_list, action_list):
        obsers_all = np.concatenate([obsers_all, obser[np.newaxis, :]], axis=0)
        actions_all = np.concatenate([actions_all, np.array([act])[np.newaxis, :]], axis=0)
    obsers_all = obsers_all[1:, :]
    actions_all = actions_all[1:, :].astype(int)
    actions_all = one_hot_encoding_numpy(actions_all.ravel().tolist(), 4)

    # # Training an initialized policy.
    net_s = tf.placeholder(tf.float32, [None, n_features], name='net_s')  # observations
    net_a = tf.placeholder(tf.int32, [None, n_actions], name='net_a')  # expert actions

    net_l1 = tf.contrib.layers.fully_connected(net_s, 32, activation_fn=tf.nn.relu)
    # net_l2 = tf.contrib.layers.fully_connected(net_l1, 128, activation_fn=tf.nn.relu)
    net_out = tf.contrib.layers.fully_connected(net_l1, n_actions, activation_fn=tf.nn.softmax)

    net_loss = tf.losses.softmax_cross_entropy(onehot_labels=net_a, logits=net_out)  # compute cost
    train_op = tf.train.AdamOptimizer(0.001).minimize(net_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if restore_path is not None:
            saver.restore(sess, restore_path)
            print("Model restore in path: {}".format(restore_path))
        else:
            print("No pretrained model found.")

        for step in range(TRAIN_STEPS):
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

        save_path = saver.save(sess, BASE_LOGS + 'model_init.ckpt')
        print("Model saved in path: {}".format(save_path))

    # # Dataset Aggregation and Retraining the policy.
    for i in range(dagger_itr):
        restore_path = save_path
        # restore_path = BASE_LOGS + 'model_init.ckpt'

        obser_list = []
        action_list = []
        obsers_new = np.zeros((1, n_features))
        actions_new = np.zeros((1, 1))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if restore_path is not None:
                saver.restore(sess, restore_path)
                print("Model restore in path: {}".format(restore_path))
            else:
                raise Exception("No pretrained model found.")

            for j in range(RUN_STEPS):
                episode_step = 0
                s = env.reset()
                while True:
                    env.render(RENDER_TIME)
                    a = sess.run(net_out, {net_s: s.ravel()[np.newaxis, :]})
                    a = np.argmax(a, axis=1)[0]
                    obser_list.append(s.ravel())
                    action_list.append(get_expert_action(s))  # add action what the expert teaches.
                    s_, r, done, info = env.step(a)  # act action what the model output.

                    s = s_
                    episode_step += 1

                    if episode_step > 299:
                        done = True

                    if done:
                        if info == 'running':
                            episode_step = 300
                            success_holder.append(0)
                        elif info == 'terminal':
                            if episode_step < 50:
                                success_holder.append(1)
                            else:
                                success_holder.append(1)
                        else:
                            raise Exception("Invalid info code.")
                        env.render(RENDER_TIME)
                        episode_step_holder.append(episode_step)
                        break

            assert len(obser_list) == len(action_list)
            for obser, act in zip(obser_list, action_list):
                obsers_new = np.concatenate([obsers_new, obser[np.newaxis, :]], axis=0)
                actions_new = np.concatenate([actions_new, np.array([act])[np.newaxis, :]], axis=0)
            obsers_new = obsers_new[1:, :]
            actions_new = actions_new[1:, :].astype(int)
            actions_new = one_hot_encoding_numpy(actions_new.ravel().tolist(), 4)
            # Dataset Aggregation
            obsers_all = np.concatenate([obsers_all, obsers_new], axis=0)
            actions_all = np.concatenate([actions_all, actions_new], axis=0)
            # Retraining the policy
            saver = tf.train.Saver()
            saver.restore(sess, restore_path)
            print("Model restore in path: {}".format(restore_path))

            for step in range(TRAIN_STEPS):
                b_s, b_a = get_batch(obsers_all, actions_all, batch_size=32)
                _, loss_ = sess.run([train_op, net_loss], {net_s: b_s, net_a: b_a})

                if step % 5 == 0:
                    output = sess.run(net_out, {net_s: obsers_all, net_a: actions_all})
                    n_accuracy = np.where(np.equal(np.argmax(output, axis=1), np.argmax(actions_all, axis=1)))[0]
                    accuracy_ = n_accuracy.shape[0] / actions_all.shape[0]
                    print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

            output = sess.run(net_out, {net_s: obsers_all, net_a: actions_all})
            n_accuracy = np.where(np.equal(np.argmax(output, axis=1), np.argmax(actions_all, axis=1)))[0]
            accuracy_ = n_accuracy.shape[0] / actions_all.shape[0]
            print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)
            print("Size of the dataset {0}".format(obsers_all.shape[0]))

            save_path = saver.save(sess, BASE_LOGS + 'model_{}.ckpt'.format(i))
            print("Model saved in path: {}".format(save_path))

    plot_rate(success_holder, './logs/dagger/model/', index=0)

    # # Testing the final policy.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if restore_path is not None:
            saver.restore(sess, restore_path)
            print("Model restore in path: {}".format(restore_path))
        else:
            raise Exception("No pretrained model found.")

        for epi in range(100):
            s = env.reset()
            while True:
                env.render(0)
                a = sess.run(net_out, {net_s: s.ravel()[np.newaxis, :]})
                a = np.argmax(a, axis=1)[0]
                s_, r, done, info = env.step(a)  # act action what the model output.
                if done:
                    env.render(0)
                    break
                s = s_

    # # destroy the env.
    env.destroy()


def enter_dagger():
    global env
    env = Maze('./env/maps/map3.json')
    env.after(100, run_maze)
    env.mainloop()  # mainloop() to run the application.


def main():
    global env
    env = Maze('./env/maps/map3.json')
    env.after(100, run_maze)
    env.mainloop()  # mainloop() to run the application.


if __name__ == '__main__':
    main()
