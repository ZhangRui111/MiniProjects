import json
import numpy as np
import time
import tkinter as tk


class Maze(tk.Tk, object):
    def __init__(self, path_to_map):
        super(Maze, self).__init__()

        map = self.read_map(path_to_map)
        self.height = map['height']
        self.width = map['width']
        self.obstacles_origin = map['obstacles']  # all obstacles' position.
        self.exit_origin = map['exit']  # exit's position.
        self.player_origin = map['player']  # player's position.

        self.unit = 40  # size of one self.unit/pixels.
        self.action_space = ['u', 'd', 'l', 'r']  # four actions: up, down, left, right.
        self.n_actions = len(self.action_space)

        self.title('Gui Maze')
        self.geometry('{0}x{1}'.format(self.height * self.unit, self.height * self.unit))  # windows background.
        self._build_maze()

    def read_map(self, path_to_map):
        with open(path_to_map) as json_file:
            data = json.load(json_file)
            return data

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=self.height * self.unit, width=self.width * self.unit)  # maze.

        # create grids.
        for c in range(0, self.width * self.unit, self.unit):
            x0, y0, x1, y1 = c, 0, c, self.height * self.unit
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.height * self.unit, self.unit):
            x0, y0, x1, y1 = 0, r, self.width * self.unit, r
            self.canvas.create_line(x0, y0, x1, y1)

        # coordination of the origin point (center of the top left corner unit).
        self.origin = np.array([self.unit/2, self.unit/2])

        # create obstacle (gray rectangle).
        self.obstacles_coords = []  # hold all obstacles' coordination.
        for i in range(len(self.obstacles_origin)):
            obstacle_center = self.origin + np.array(
                [self.unit * self.obstacles_origin[i][0], self.unit * self.obstacles_origin[i][1]])
            obstacle = self.canvas.create_rectangle(
                obstacle_center[0] - 15, obstacle_center[1] - 15,
                obstacle_center[0] + 15, obstacle_center[1] + 15,
                fill='gray')
            self.obstacles_coords.append(self.canvas.coords(obstacle))

        # create exit (green oval)
        exit_center = self.origin + np.array(
            [self.unit * self.exit_origin[0][0], self.unit * self.exit_origin[0][1]])
        self.exit = self.canvas.create_oval(
            exit_center[0] - 15, exit_center[1] - 15,
            exit_center[0] + 15, exit_center[1] + 15,
            fill='green')

        # create player (yellow oval)
        player_center = self.origin + np.array(
            [self.unit * self.player_origin[0][0], self.unit * self.player_origin[0][1]])
        self.player = self.canvas.create_oval(
            player_center[0] - 15, player_center[1] - 15,
            player_center[0] + 15, player_center[1] + 15,
            fill='yellow')

        # pack all
        self.canvas.pack()

        # pixels value
        # img = tk.PhotoImage(height=self.height * self.unit, width=self.width * self.unit)
        # self.canvas.create_image((self.height * self.unit, self.height * self.unit), image=img, state="normal")

    def reset(self):
        self.update()
        self.canvas.delete(self.player)
        # reset the player.
        player_center = self.origin + np.array(
            [self.unit * self.player_origin[0][0], self.unit * self.player_origin[0][1]])
        self.player = self.canvas.create_oval(
            player_center[0] - 15, player_center[1] - 15,
            player_center[0] + 15, player_center[1] + 15,
            fill='yellow')
        # return observation
        return self.canvas.coords(self.player)

    def step(self, action):
        s = self.canvas.coords(self.player)  # player's current coordination.
        base_action = np.array([0, 0])  # guide the player's real movement.
        s_est = s.copy()  # estimation of player's next coordination.
        # # ------- verify s_est == s_real ------- # #
        # if_cmp = True
        # # ------- verify s_est == s_real ------- # #
        if action == 0:  # up
            if s[1] > self.unit:
                base_action[1] -= self.unit
                s_est[1] -= 40.
                s_est[3] -= 40.
        elif action == 1:  # down
            if s[1] < (self.height - 1) * self.unit:
                base_action[1] += self.unit
                s_est[1] += 40.
                s_est[3] += 40.
        elif action == 2:  # left
            if s[0] > self.unit:
                base_action[0] -= self.unit
                s_est[0] -= 40.
                s_est[2] -= 40.
        elif action == 3:  # right
            if s[0] < (self.width - 1) * self.unit:
                base_action[0] += self.unit
                s_est[0] += 40.
                s_est[2] += 40.
        else:
            print('illegal action!')

        # self.canvas.move(self.player, base_action[0], base_action[1])  # move agent
        # s_real = self.canvas.coords(self.player)  # next state

        # reward function
        if s_est == self.canvas.coords(self.exit):
            reward = 1
            done = True
            info = 'terminal'
        else:
            if s_est in self.obstacles_coords:
                base_action = np.array([0, 0])
                # # ------- verify s_est == s_real ------- # #
                # if_cmp = False
                # # ------- verify s_est == s_real ------- # #
            reward = -1
            done = False
            info = 'running'

        self.canvas.move(self.player, base_action[0], base_action[1])  # move agent
        s_real = self.canvas.coords(self.player)  # player's real next coordination.
        # # ------- verify s_est == s_real ------- # #
        # if if_cmp:
        #     s_est = [int(x) for x in s_est]
        #     s_real = [int(x) for x in s_real]
        #     assert s_est == s_real
        # # ------- verify s_est == s_real ------- # #

        return s_real, reward, done, info

    def render(self, slt=None):
        if slt is not None:
            time.sleep(slt)
        self.update()


# def my_update():
#     for t in range(10):
#         steps = 0
#         env.reset()
#         while True:
#             env.render(0)
#             a = np.random.random_integers(4)-1
#             s, r, done, info = env.step(a)
#             steps += 1
#             # print('action:{0} | reward:{1} | done: {2}'.format(a, r, done))
#             if done:
#                 print('{0} -- {1}'.format(info, steps))
#                 env.render(0)
#                 break
#
#
# def main():
#     global env
#     env = Maze('./maps/map2.json')
#     env.after(100, my_update)  # Call function update() once after given time/ms.
#     env.mainloop()  # mainloop() to run the application.
#
#
# if __name__ == '__main__':
#     main()
