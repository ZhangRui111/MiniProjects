import numpy as np
import tensorflow as tf

# np.random.seed(1)
# tf.set_random_seed(1)


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            restore_path=None,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=64,
            e_greedy_init=0,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = e_greedy_init if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        if output_graph:
            tf.summary.FileWriter('./logs/double_dqn/', self.sess.graph)

        if restore_path is not None:
            vars_list = [v for v in tf.trainable_variables()]
            print(len(vars_list))
            print(vars_list)
            restore_vars_dict_e = {
                "net/l1/fully_connected/weights": vars_list[0],
                "net/l1/fully_connected/biases": vars_list[1],
                "net/l2/fully_connected/weights": vars_list[2],
                "net/l2/fully_connected/biases": vars_list[3],
            }
            saver_e = tf.train.Saver(restore_vars_dict_e)
            saver_e.restore(self.sess, restore_path)
            # restore weights under the `target_net` namescope [optional].
            restore_vars_dict_t = {
                "net/l1/fully_connected/weights": vars_list[4],
                "net/l1/fully_connected/biases": vars_list[5],
                "net/l2/fully_connected/weights": vars_list[6],
                "net/l2/fully_connected/biases": vars_list[7],
            }
            saver_t = tf.train.Saver(restore_vars_dict_t)
            saver_t.restore(self.sess, restore_path)
            print("Model restore in path: {}".format(restore_path))
        else:
            print("No pretrained model found.")

        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            with tf.variable_scope('l1'):
                l1 = tf.contrib.layers.fully_connected(self.s, 32, activation_fn=tf.nn.relu)
            with tf.variable_scope('l2'):
                self.q_eval = tf.contrib.layers.fully_connected(l1, self.n_actions, activation_fn=tf.nn.softmax)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            with tf.variable_scope('l1'):
                l1 = tf.contrib.layers.fully_connected(self.s_, 32, activation_fn=tf.nn.relu)
            with tf.variable_scope('l2'):
                self.q_next = tf.contrib.layers.fully_connected(l1, self.n_actions, activation_fn=tf.nn.softmax)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.n_actions)
        else:
            observation = observation[np.newaxis, :]
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        return action

    def learn(self, done):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        # max_act4next = np.argmax(q_eval4next, axis=1)
        selected_q_next = np.max(q_next, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        if done:
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
