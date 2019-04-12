import numpy as np
import time

from utils import plot_steps, plot_multi_lines, running_average
from env.maze import Maze

MAX_EPISODES = 500
N = 25  # planning steps after every real step.


def update_dyna_q(n):
    steps_episodes = []
    for episode in range(MAX_EPISODES):
        steps = 0
        s = env.reset()
        while True:
            # env.render()
            a = RL.choose_action(str(s))
            s_, r, done, info = env.step(a)
            steps += 1
            RL.learn(str(s), a, r, str(s_))
            env_model.store_transition(str(s), a, r, s_)
            s = s_

            for i in range(n):     # learn 10 more times using the env_model
                ms, ma = env_model.sample_s_a()  # ms in here is a str
                mr, ms_ = env_model.get_r_s_(ms, ma)
                RL.learn(ms, ma, mr, str(ms_))

            if done:
                print('{0} -- {1}'.format(episode, steps))
                steps_episodes.append(steps)
                # env.render(1)
                break

    # end of game
    print(RL.q_table)
    # plot_steps(steps_episodes, save_path='./logs/dyna_q/dyna_q.png')
    average_steps = running_average(steps_episodes, interval=50)
    plot_steps(average_steps, save_path='./logs/dyna_q/dyna_q_{}.png'.format(n))
    np.save('./logs/dyna_q/dyna_q_{}.npy'.format(n), np.asarray(average_steps))
    env.destroy()
    print('game over')


def update_q():
    steps_episodes = []
    for episode in range(MAX_EPISODES):
        steps = 0
        s = env.reset()
        while True:
            # env.render()
            a = RL.choose_action(str(s))
            s_, r, done, info = env.step(a)
            steps += 1
            RL.learn(str(s), a, r, str(s_))
            s = s_

            if done:
                print('{0} -- {1}'.format(episode, steps))
                steps_episodes.append(steps)
                # env.render(1)
                break

    # end of game
    print(RL.q_table)
    # plot_steps(steps_episodes, save_path='./logs/q_learning/q_learning.png')
    average_steps = running_average(steps_episodes, interval=50)
    plot_steps(average_steps, save_path='./logs/q_learning/q_learning.png')
    np.save('./logs/q_learning/q_learning.npy', np.asarray(average_steps))
    env.destroy()
    print('game over')


def main():
    global env, RL, env_model

    # if_dyna = True
    # env = Maze('./env/maps/map2.json')
    # if if_dyna:
    #     # ---------- Dyna Q ---------- # #
    #     from brain.dyna_Q import QLearningTable, EnvModel
    #     RL = QLearningTable(actions=list(range(env.n_actions)))
    #     env_model = EnvModel(actions=list(range(env.n_actions)))
    #     env.after(0, update_dyna_q)  # Call function update() once after given time/ms.
    # else:
    #     # # -------- Q Learning -------- # #
    #     from brain.Q_learning import QLearningTable
    #     RL = QLearningTable(actions=list(range(env.n_actions)))
    #     env.after(0, update_q())  # Call function update() once after given time/ms.

    time_cmp = []
    # -------- Q Learning -------- # #
    from brain.Q_learning import QLearningTable
    start = time.time()
    env = Maze('./env/maps/map2.json')
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(0, update_q())  # Call function update() once after given time/ms.
    env.mainloop()
    sum_time = time.time()-start
    time_cmp.append(sum_time)
    # ---------- Dyna Q ---------- # #
    from brain.dyna_Q import QLearningTable, EnvModel
    for n in [5, 10, 25, 50]:
        start = time.time()
        env = Maze('./env/maps/map2.json')
        RL = QLearningTable(actions=list(range(env.n_actions)))
        env_model = EnvModel(actions=list(range(env.n_actions)))
        print('Dyna-{}'.format(n))
        env.after(0, update_dyna_q, n)  # n is the parameter of update_dyna_q().
        env.mainloop()  # mainloop() to run the application.
        sum_time = time.time() - start
        time_cmp.append(sum_time)

    # This part must after env.mainloop()
    # plot all lines.
    all_aver_steps = [np.load('./logs/q_learning/q_learning.npy').tolist()]
    for n in [5, 10, 25, 50]:
        all_aver_steps.append(np.load('./logs/dyna_q/dyna_q_{}.npy'.format(n)).tolist())
    plot_multi_lines(all_aver_steps,
                     all_labels=['q_learning', 'dyna_5', 'dyna_10', 'dyna_25', 'dyna_50'],
                     save_path='./logs/cmp_all.png')

    # only plot dyna_Q
    all_aver_steps = []
    for n in [5, 10, 25, 50]:
        all_aver_steps.append(np.load('./logs/dyna_q/dyna_q_{}.npy'.format(n))[0:100].tolist())
    plot_multi_lines(all_aver_steps,
                     all_labels=['dyna_5', 'dyna_10', 'dyna_25', 'dyna_50'],
                     save_path='./logs/cmp_all_dyna_Q.png')

    print(time_cmp)


if __name__ == '__main__':
    main()

