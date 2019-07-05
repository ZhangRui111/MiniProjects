"""
Restore weights from pretrained model to build/assemble a bigger new model.
For example, if you have trained a two-layer model before you train a three-layer model.
We can restore the weights from the two-layer model to initialize the first two layers
of the three-layer model.
"""
import numpy as np
import tensorflow as tf

from env.maze import Maze


def build_net(n_actions, n_features):
    # ------------------ build evaluate_net ------------------
    s = tf.placeholder(tf.float32, [None, n_features], name='s')  # input
    s_ = tf.placeholder(tf.float32, [None, n_features], name='s_')  # input
    q_target = tf.placeholder(tf.float32, [None, n_actions], name='Q_target')  # for calculating loss

    with tf.variable_scope('eval_net'):
        with tf.variable_scope('l1'):
            l1 = tf.contrib.layers.fully_connected(s, 32, activation_fn=tf.nn.relu)
        with tf.variable_scope('l2'):
            q_eval = tf.contrib.layers.fully_connected(l1, n_actions, activation_fn=tf.nn.softmax)
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.squared_difference(q_target, q_eval))
    with tf.variable_scope('train'):
        _train_op = tf.train.RMSPropOptimizer(0.0001).minimize(loss)

    # ------------------ build target_net ------------------
    with tf.variable_scope('target_net'):
        with tf.variable_scope('l1'):
            l1 = tf.contrib.layers.fully_connected(s_, 32, activation_fn=tf.nn.relu)
        with tf.variable_scope('l2'):
            q_next = tf.contrib.layers.fully_connected(l1, n_actions, activation_fn=tf.nn.softmax)

    return [[s, s_, q_target], [q_eval, loss, _train_op, q_next]]


def run_maze():
    network = build_net(4, env.height*env.width)
    # network: [[s, s_, q_target], [q_eval, loss, _train_op, q_next]]
    net_s = network[0][0]
    net_out = network[1][0]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # # ----------------------- 1 ----------------------- # #
        # Get all the variables in the graph as a list.         #
        # # ----------------------- 1 ----------------------- # #
        # vars_list = [v for v in tf.global_variables()]
        vars_list = [v for v in tf.trainable_variables()]
        print(vars_list)
        print(vars_list[0].name)
        print(vars_list[0].shape)
        print(vars_list[0].trainable)
        # # ----------------------- 1 ----------------------- # #

        # # ----------------------- 5 ----------------------- # #
        # saver.restore() to restore all the variables.         #
        # All variables' name or other graph keys must be found #
        # from the checkpoint.                                  #
        # # ----------------------- 5 ----------------------- # #
        # saver = tf.train.Saver()
        # saver.restore(sess, './logs/model_1/model_init.ckpt')
        # # ----------------------- 5 ----------------------- # #
        # print("Model restore in path: {}".format('./logs/model_1/model_init.ckpt'))

        # # ----------------------- 6 ----------------------- # #
        # Restore part of the variables.                        #
        #                                                       #
        # passing in a Python dictionary, that maps string keys #
        # (in the checkpoint file) to tf.Variable objects (in   #
        # the target graph):                                    #
        # saver = tf.train.Saver({"v2": v1}), that means,       #
        # you want the v1 variable in your graph to             #
        # be restored via the name "v2".                        #
        #                                                       #
        # To restore them, we need to remove the special "tail" #
        # of the variable names.                                #
        # For example,                                          #
        # tf.Variable 'v1:0' gives the name as 'v1:0'.          #
        # When you have name scopes, you might have many v1     #
        # but with different names before it: 'foobar/v1:0'.    #
        # If you want to restore the first v1, then the key of  #
        # the dictionary you pass into the Saver function will  #
        # be: 'foobar/v1', so we apply something like           #
        # v.name[:-2] here to avoid the last two chars.         #
        # # ----------------------- 6 ----------------------- # #

        # restore weights under the `eval_net` namescope [optional].
        restore_vars_dict_e = {
            "net/l1/fully_connected/weights": vars_list[0],
            "net/l1/fully_connected/biases": vars_list[1],
            "net/l2/fully_connected/weights": vars_list[2],
            "net/l2/fully_connected/biases": vars_list[3],
        }
        saver_e = tf.train.Saver(restore_vars_dict_e)
        saver_e.restore(sess, './logs/model_1/model_init.ckpt')
        # restore weights under the `target_net` namescope [optional].
        restore_vars_dict_t = {
            "net/l1/fully_connected/weights": vars_list[4],
            "net/l1/fully_connected/biases": vars_list[5],
            "net/l2/fully_connected/weights": vars_list[6],
            "net/l2/fully_connected/biases": vars_list[7],
        }
        saver_t = tf.train.Saver(restore_vars_dict_t)
        saver_t.restore(sess, './logs/model_1/model_init.ckpt')
        # # ----------------------- 6 ----------------------- # #

        # # ----------------------- 7 ----------------------- # #
        # Get the weights of the current model and check        #
        # whether we have successfully restore the weights      #
        # from a pretrained model.                              #
        # # ----------------------- 7 ----------------------- # #
        train_vars = tf.trainable_variables()
        print(train_vars)
        print(len(train_vars))

        eval_l1_weights = sess.run(train_vars[0])
        target_l1_weights = sess.run(train_vars[4])
        a = np.equal(eval_l1_weights, target_l1_weights)
        assert np.array_equal(eval_l1_weights, target_l1_weights) is True
        print(eval_l1_weights.shape)
        print(target_l1_weights.shape)
        print(eval_l1_weights[0])
        print(target_l1_weights[0])
        # # ----------------------- 7 ----------------------- # #

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
    print('Destroy the env.')
    env.destroy()


def main():
    global env
    env = Maze('./env/maps/map3.json')
    env.after(100, run_maze)
    env.mainloop()  # mainloop() to run the application.


if __name__ == '__main__':
    main()
