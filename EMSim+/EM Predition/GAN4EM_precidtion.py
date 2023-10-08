#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, \
    Concatenate
from tensorflow.keras import Model, regularizers
import numpy as np
import matplotlib.pyplot as plt
from time import time
import matplotlib.animation as animation
import os
from generator import Generator
from discriminator import Discriminator
import shutil
from Logger import Logger
import sys


def getToolLogo():
    logo = [
        '====================================================================================',
        '        ______ __  __  _____ _                                                    	     ',
        '        |  ____|  \/  |/ ____(_)           _                                      	     ',
        '	| |__  | \  / | (___  _ _ __ ___ _| |_ 					                         ',
        '	|  __| | |\/| |\___ \| | \_ ` _ \_   _|					                         ',
        '	| |____| |  | |____) | | | | | | ||_|                			                 ',
        '	|______|_|  |_|_____/|_|_| |_| |_|						                         ',
        ' 											                                         ',
        '====================================================================================',
    ]
    return '\n'.join(logo) + '\n'


# 保存运行过程的控制台信息
sys.stdout = Logger("Default.txt", sys.stdout)

num_probe_x_tiles = 48
num_probe_y_tiles = 48
time_steps = 20
percent_valid_split = 0.9  # 10 percent of training points is used for validation during training
learning_rate_val = 0.0005
num_input_stimuli = 250
decay_rate_val = 0.98
#epochs = 100
#batch_size = 64


class CGAN():
    def __init__(self):
        initial_learning_rate = learning_rate_val
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=num_input_stimuli,
            decay_rate=decay_rate_val,
            staircase=True)

        self.Discriminator = Discriminator()
        self.Discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mse',
                                   metrics=['accuracy'])

        # Build the generator
        self.Generator = Generator()
        y_data = Input(shape=(num_probe_x_tiles, num_probe_y_tiles, 1))
        x_data = Input(shape=(num_probe_x_tiles, num_probe_y_tiles, 2))
        t_data = Input(shape=1)
        f_data = self.Generator([x_data, t_data])

        # The discriminator takes generated image as input
        # and determines validity and the label of that image
        valid = self.Discriminator([x_data, t_data, f_data])

        # For the combined model we will only train the generator
        self.Discriminator.trainable = False

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.Combined = Model(inputs=[y_data, x_data, t_data], outputs=[valid, f_data])
        self.Combined.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=['mse', 'mae'],
                              loss_weights=[1, 100], metrics=['mse', 'mae', 'mape'])


    def EM_prediction(self, normalization_data, current_map_data, grid_map_data, em_map_data, model_path):
        current_map = np.load(current_map_data)
        grid_map = np.load(grid_map_data)
        em_map = np.load(em_map_data)

        data_runs = np.shape(current_map)[0]

        self.Generator.load_weights(model_path)
        max_x = normalization_data[0]
        min_x = normalization_data[1]
        max_g = normalization_data[2]
        min_g = normalization_data[3]
        max_t = normalization_data[4]
        min_t = normalization_data[5]
        max_y = normalization_data[6]
        min_y = normalization_data[7]
        Writer = animation.writers['html']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        im = []
        figs = []
        anis = []
        num_im = data_runs
        predicted_video = np.zeros((num_im, time_steps, num_probe_x_tiles, num_probe_y_tiles))
        err_video = np.zeros((num_im, time_steps, num_probe_x_tiles, num_probe_y_tiles))
        for im_num in range(num_im):
            def updatefig1(frame, im_num):
                im[im_num * 3].set_array(em_map[im_num, frame, ...])
                im[im_num * 3 + 1].set_array(predicted_video[im_num, frame, ...])

            def updatefig2(frame, im_num):
                im[im_num * 3 + 2].set_array(err_video[im_num, frame, ...])

            for frame in range(time_steps):
                data_pt = np.array([[frame]]) / max_t
                in_data = current_map[im_num:im_num + 1, frame, ..., np.newaxis]
                in_data = (in_data - min_x) / (max_x - min_x)
                g_data = grid_map[np.newaxis, ..., np.newaxis]
                g_data = (g_data - min_g) / (max_g - min_g)
                in_data = np.concatenate((in_data, g_data), axis=-1)
                predicted = self.Generator.predict((in_data, data_pt))
                predicted_video[im_num, frame, ...] = (np.squeeze(predicted) * (max_y - min_y)) + min_y
                # predicted_video[im_num, frame, ...] = np.squeeze(predicted)
                err_video[im_num, frame, ...] = predicted_video[im_num, frame, ...] - em_map[im_num, frame, ...]
            print("Writing result of test: ", im_num)
            test_data_num = im_num
            fig, axes = plt.subplots(1, 2)
            max_val = np.max((np.max(em_map[im_num, ...]), np.max(predicted_video[im_num, ...])))
            max_err = np.max(err_video[im_num, ...])
            min_err = np.min(err_video[im_num, ...])
            im.append(axes[0].imshow(em_map[im_num, 0], vmin=0, vmax=max_val))
            im.append(axes[1].imshow(predicted_video[im_num, 0, ...], vmin=0, vmax=max_val))
            fig.colorbar(im[3 * im_num + 1], ax=axes.ravel().tolist())

            anis.append(
                animation.FuncAnimation(fig, updatefig1, frames=range(time_steps), fargs=(im_num,), interval=150))
            anis[2 * im_num].save("output_plots/run_contours_%d.html" % test_data_num, writer=writer)
            plt.close()
            fig2 = plt.figure()
            im.append(plt.imshow(err_video[im_num, 0, ...], vmin=min_err, vmax=max_err))
            plt.colorbar()
            anis.append(
                animation.FuncAnimation(fig2, updatefig2, frames=range(time_steps), fargs=(im_num,), interval=150))
            anis[2 * im_num + 1].save("output_plots/err_contours_%d.html" % test_data_num, writer=writer)
            plt.close()
            dir = os.path.join("./", "output_plots", "output_plots")
            if not os.path.exists(dir):
                os.mkdir(dir)
                os.rename("output_plots/run_contours_%d_frames" % test_data_num,
                          "output_plots/output_plots/run_contours_%d_frames" % test_data_num)
                os.rename("output_plots/err_contours_%d_frames" % test_data_num,
                          "output_plots/output_plots/err_contours_%d_frames" % test_data_num)
            else:
                if os.path.exists(
                        "output_plots/output_plots/run_contours_%d_frames" % test_data_num) and os.path.exists(
                        "output_plots/output_plots/err_contours_%d_frames" % test_data_num):
                    shutil.rmtree("output_plots/output_plots/run_contours_%d_frames" % test_data_num)
                    shutil.rmtree("output_plots/output_plots/err_contours_%d_frames" % test_data_num)
                os.rename("output_plots/run_contours_%d_frames" % test_data_num,
                          "output_plots/output_plots/run_contours_%d_frames" % test_data_num)
                os.rename("output_plots/err_contours_%d_frames" % test_data_num,
                          "output_plots/output_plots/err_contours_%d_frames" % test_data_num)

        np.save('Training_EM_prediction.npy', predicted_video)
        np.save('Err_EM_prediction.npy', err_video)

    def read_data(self, current_map_data, grid_map_data, em_map_data):
        current_map = np.load(current_map_data)
        grid_map = np.load(grid_map_data)
        em_map = np.load(em_map_data)

        data_runs = np.shape(current_map)[0]
        num_images = data_runs * time_steps

        count = 0
        x_data = np.zeros((num_images, num_probe_x_tiles, num_probe_y_tiles, 1))
        g_data = np.zeros((num_images, num_probe_x_tiles, num_probe_y_tiles, 1))
        y_data = np.zeros((num_images, num_probe_x_tiles, num_probe_y_tiles, 1))
        t_data = np.zeros((num_images, 1))
        for im_num in range(data_runs):
            for frame in range(time_steps):
                x_data[count, :, :, 0] = current_map[im_num, frame, ...]
                g_data[count, :, :, 0] = grid_map
                y_data[count, :, :, 0] = em_map[im_num, frame, ...]
                t_data[count, ...] = frame
                count += 1
        return [x_data, g_data, y_data, t_data]

    def process_data(self, x_data, g_data, y_data, t_data):
        indices = np.arange(x_data.shape[0])
        np.random.shuffle(indices)

        x_data = x_data[indices]
        g_data = g_data[indices]
        y_data = y_data[indices]
        t_data = t_data[indices]

        min_x = np.min(x_data)
        max_x = np.max(x_data)
        x_data = (x_data - min_x) / (max_x - min_x)

        min_g = np.min(g_data)
        max_g = np.max(g_data)
        g_data = (g_data - min_g) / (max_g - min_g)

        x_data = np.concatenate((x_data, g_data), axis=-1)

        max_t = time_steps - 1
        min_t = 0
        t_data = t_data / max_t

        min_y = np.min(y_data)
        max_y = np.max(y_data)
        y_data = (y_data - min_y) / (max_y - min_y)
        normalization_data = [max_x, min_x, max_g, min_g, max_t, min_t, max_y, min_y]
        return x_data, y_data, t_data, normalization_data


if __name__ == '__main__':
    start_time = time()  # obtain the starting time of code.


    # Model save path
    trained_model_path = '../GAN Model Training/weights/generator_trained_weights.h5'

    # load training dataset
    test_current_map = 'current_map_test.npy'
    test_grid_map = 'power_grid_map.npy'
    test_em_map = 'em_map_test.npy'

    # exit()
    ##############################Pre-training Stage#####################################
    print('Step 1: Build the cGAN model')
    cgan = CGAN()

    print('Step 2: Load test Dataset')
    data_train = cgan.read_data(test_current_map, test_grid_map, test_em_map)

    print('Step 3: Pre-process Dataset')
    x_data, y_data, t_data, normalization_data = cgan.process_data(data_train[0], data_train[1], data_train[2], data_train[3])

    print('Step 4: EM prediction and evaluate the trained GAN model')

    cgan.EM_prediction(normalization_data, test_current_map, test_grid_map, test_em_map, trained_model_path)

    print("\nTotal running time of pre-training stage: --- %s seconds ---\n" % (time() - start_time))  # obtain the total running time of code
    # exit()

# print(file = outputfile)
# outputfile.close() # close后才能看到写入的数据
