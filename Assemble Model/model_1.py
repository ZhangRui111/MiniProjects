"""
Save the initialized model for further use/assemble.
This example is illustrated by DAgger algorithm.
"""
import numpy as np
import tensorflow as tf

from env.maze import Maze


# Stimulated expert library.
ACTION_UP = [[4, 2], [4, 3], [5, 0], [5, 2]]
ACTION_DOWN = [[0, 0], [0, 1], [0, 4], [1, 1], [1, 4], [2, 2], [2, 4], [3, 5], [4, 5]]
ACTION_LEFT = [[0, 2], [0, 5]]
ACTION_RIGHT = [[1, 0], [2, 1], [3, 2], [3, 3], [3, 4], [4, 0], [4, 1]]

RUN_STEPS = 200
TRAIN_STEPS = 100
RENDER_TIME = 0
BASE_LOGS = './logs/model_1/'


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


def get_batch(X, y, batch_size):
    """ Get one batch data. """
    data_size = X.shape[0]
    rand = np.random.random_integers(0, data_size, 1)[0]
    rand = min(rand, data_size-batch_size)
    return X[rand:rand + batch_size, :], y[rand:rand + batch_size, :]


def one_hot_encoding_numpy(array_list, size):
    """
    One Hot Encoding using numpy.
    :param array_list: i.e., [1, 2, 3]
    :param size: one hot size, i.e., 4
    :return: ndarray i.e., [[0 1 0 0]
                            [0 0 1 0]
                            [0 0 0 1]]
    """
    one_hot_array = np.eye(size)[array_list].astype(int)
    return one_hot_array


def run_maze():
    n_features = env.height * env.width
    n_actions = 4

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

    with tf.variable_scope('net'):
        with tf.variable_scope('l1'):
            net_l1 = tf.contrib.layers.fully_connected(net_s, 32, activation_fn=tf.nn.relu)
            # net_l2 = tf.contrib.layers.fully_connected(net_l1, 128, activation_fn=tf.nn.relu)
        with tf.variable_scope('l2'):
            net_out = tf.contrib.layers.fully_connected(net_l1, n_actions, activation_fn=tf.nn.softmax)

    with tf.variable_scope('loss'):
        net_loss = tf.losses.softmax_cross_entropy(onehot_labels=net_a, logits=net_out)  # compute cost
    with tf.variable_scope('train'):
        train_op = tf.train.AdamOptimizer(0.001).minimize(net_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

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

        # # ----------------------- 1 ----------------------- # #
        # Get all the variables in the graph as a list.         #
        # # ----------------------- 1 ----------------------- # #
        vars_list = [v for v in tf.global_variables()]
        print(vars_list)
        print(vars_list[0].name)
        print(vars_list[0].shape)
        print(vars_list[0].trainable)
        # # ----------------------- 1 ----------------------- # #

        # # ----------------------- 2 ----------------------- # #
        # saver.save() to save all the variables.               #
        # # ----------------------- 2 ----------------------- # #
        saver = tf.train.Saver()
        save_path = saver.save(sess, BASE_LOGS + 'model_init.ckpt')
        # # ----------------------- 2 ----------------------- # #
        print("Model saved in path: {}".format(save_path))

        # # ----------------------- 3 ----------------------- # #
        # Pass a variable list or a dictionary into the
        # function, and only these will be saved.               #
        # # ----------------------- 3 ----------------------- # #
        saved_vars_list = [vars_list[0], vars_list[1]]
        saver_part = tf.train.Saver(saved_vars_list)
        save_path_part = saver_part.save(sess, BASE_LOGS + 'model_init_part.ckpt')
        # # ----------------------- 3 ----------------------- # #
        print("Model saved in path: {}".format(save_path_part))

        # # ----------------------- 4 ----------------------- # #
        # Print all variables in a saved model and check if     #
        # a certain variable has been saved in the model.       #
        # # ----------------------- 4 ----------------------- # #
        # model_path = BASE_LOGS + 'model_init.ckpt'
        model_path = BASE_LOGS + 'model_init_part.ckpt'
        from tensorflow.python.tools import inspect_checkpoint as chkp
        # print all tensors in checkpoint file.
        chkp.print_tensors_in_checkpoint_file(model_path, tensor_name='', all_tensors=True)
        # check a certain variable.
        from tensorflow.python import pywrap_tensorflow
        reader = pywrap_tensorflow.NewCheckpointReader(model_path)
        # 'var_to_shape_map' is a dictionary contains every tensor in the model
        var_to_shape_map = reader.get_variable_to_shape_map()
        if 'net/l2/fully_connected/weights:0' in var_to_shape_map.keys():
            print("Variable `net/l1/fully_connected/weights:0` is in the model.")
        else:
            print("Variable `net/l1/fully_connected/weights:0` is not in the model.")
        # # ----------------------- 4 ----------------------- # #

    # # destroy the env.
    env.destroy()


def main():
    global env
    env = Maze('./env/maps/map3.json')
    env.after(100, run_maze)
    env.mainloop()  # mainloop() to run the application.


if __name__ == '__main__':
    main()
