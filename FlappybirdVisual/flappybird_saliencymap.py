import skimage
import sys
import random
import numpy as np
import tensorflow as tf
import keras.backend as Kb

from collections import deque
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam

sys.path.append("game/")
import wrapped_flappy_bird as game
from utils import exist_delete_or_create_folder

GAME = 'flappybird'  # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVATION = 3200.  # timesteps to observe before training.
EXPLORE = 3000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

img_rows, img_cols = 80, 80
img_channels = 4  # We stack 4 frames


def build_load_model():
    print("Firstly, we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',
                            input_shape=(img_rows, img_cols, img_channels)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', name="final_conv"))
    model.add(Activation('relu', name='after'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.summary()

    print("We finish building the model. Now we load weight")
    model.load_weights("model.h5")
    model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
    print("Weight load successfully")
    return model


def dissectNetwork(model):
    mask_save_path = "./logs/mask/"
    smap_save_path = "./logs/smap/"
    exist_delete_or_create_folder(mask_save_path)
    exist_delete_or_create_folder(smap_save_path)

    # Open up a game state to communicate with emulator
    game_state = game.GameState()
    # Store the previous observations in replay memory
    D = deque()
    # Get the first state by doing nothing (1, 0) and preprocess the image to 80x80x4
    x_t, r_0, terminal = game_state.frame_step(np.array([1, 0]))

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))
    # x_t = x_t / 255.0
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    # print (s_t.shape)
    # In Keras, need to reshape to 1*80*80*4.
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

    observation = 999999999  # In this case, we keep observe, never train.
    epsilon = FINAL_EPSILON
    t = 0
    epoch = 0
    while t < observation:
        a_t = np.zeros([ACTIONS])
        q = model.predict(s_t)  # Input a stack of 4 images, get the prediction
        max_Q = np.argmax(q)
        a_t[max_Q] = 1

        # Run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1, (80, 80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        # x_t1 = x_t1 / 255.0
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        if t % 8 == 0:
            # J = Kb.gradients(model.output, model.input) * model.input  # why * model.input?
            # # J = Kb.gradients(model.output, model.input)
            # salient = sess.run(J, feed_dict={model.input: s_t1})
            # # aaa = salient * s_t1
            # # salient = salient * s_t1
            #
            # # # Testing.
            # # salient_map = np.zeros([80, 80, 3])
            # # # Scale the salient map by 100x to highlight.
            # # a = salient[:, :, :, :, -1].reshape([80, 80]) * 100
            # # # b = ((a - a.min()) * (1/(a.max() - a.min()) * 255))
            # # b = np.clip(a, 0, 255)
            # # salient_map[:, :, 0] = b
            # # masked = skimage.transform.resize(x_t1_colored, (80, 80))
            # # masked = skimage.exposure.rescale_intensity(masked, out_range=(0, 255))
            # # # masked[:, :, -1] += b
            #
            # salient_map = np.zeros([80, 80, 3])
            # # Scale the salient map by 100x to highlight.
            # salient_map[:, :, 0] = np.clip(salient[:, :, :, :, -1].reshape([80, 80]) * 100, 0, 255)
            # masked = skimage.transform.resize(x_t1_colored, (80, 80))
            # masked = skimage.exposure.rescale_intensity(masked, out_range=(0, 255))
            # # masked[:, :, -1] += np.clip(salient[:, :, :, :, -1].reshape([80, 80]) * 100, 0, 255)
            #
            # skimage.io.imsave("{0}state_{1}.jpg".format(mask_save_path, t), masked)  # masked input
            # skimage.io.imsave("{0}state_{1}.jpg".format(smap_save_path, t), salient_map)  # salient map
            # TODO: Real saliency maps implementation.

        t += 1
        s_t = s_t1

    print("Episode finished!")


def playGame():
    model = build_load_model()
    dissectNetwork(model)


def main():
    playGame()


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    Kb.set_session(sess)
    main()
