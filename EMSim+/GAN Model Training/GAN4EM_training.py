#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras import Model
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
import argparse


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
	

sys.stdout = Logger("Default.txt", sys.stdout)


class CGAN():
    def __init__(self):
        initial_learning_rate = learning_rate_val
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=num_input_stimuli,
            decay_rate=decay_rate_val,
            staircase=True)

        self.Discriminator = Discriminator()
        self.Discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mse', metrics=['accuracy'])

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
        self.Combined.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=['mse', 'mae'], loss_weights=[1, 100], metrics=['mse', 'mae', 'mape'])



    def train_model(self, x_data, t_data, y_data, model_path):
        # x_data is input image frame, t_data is time, y_data is golden output image frame
        # Adversarial ground truths
        # self.build_model(generator_trainable=True, discriminator_trainable=False)
        # self.Generator.trainable = True
        # self.Discriminator.trainable = False
        
        valid = np.ones((batch_size, 6, 6, 1))
        fake = np.zeros((batch_size, 6, 6, 1))

        data_runs = np.shape(x_data)[0]
        train_runs = int(np.round(data_runs * percent_valid_split))
        interval_sample = train_runs // batch_size
        print('interval sample: ', interval_sample)
        steps = (data_runs - train_runs) // batch_size
        print('steps: ', steps)
    
        best_loss = 0
        best_epoch = 0

        for epoch in range(epochs):
            st = time()
            for interval in range(interval_sample):
    
                #  Train Discriminator
    
                # Generate a half batch of new images
                x_data_train = x_data[batch_size*interval: batch_size*(interval+1), ...]
                t_data_train = t_data[batch_size*interval: batch_size*(interval+1), ...]
                f_data_train = self.Generator.predict([x_data_train, t_data_train])

                # Train the discriminator
                y_data_train = y_data[batch_size*interval: batch_size*(interval+1), ...]
                d_loss_real = self.Discriminator.train_on_batch([x_data_train, t_data_train, y_data_train], valid)
                print(d_loss_real)
                d_loss_fake = self.Discriminator.train_on_batch([x_data_train, t_data_train, f_data_train], fake)
                print(d_loss_fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                print(d_loss)

                # Train the generator
                g_loss = self.Combined.train_on_batch([y_data_train, x_data_train, t_data_train], [valid, y_data_train])

                # Plot the progress
                print("%d %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, interval, d_loss[0], 100 * d_loss[1], g_loss[0]))

            val_loss = 0
            for step in range(steps):
                #  Validate Generator
                x_data_valid = x_data[train_runs + batch_size*step: train_runs + batch_size*(step+1), ...]
                t_data_valid = t_data[train_runs + batch_size*step: train_runs + batch_size*(step+1), ...]
                y_data_valid = y_data[train_runs + batch_size*step: train_runs + batch_size*(step+1), ...]

                com_loss = self.Combined.evaluate([y_data_valid, x_data_valid, t_data_valid], [valid, y_data_valid])

                val_loss += com_loss[0]

            val_loss = val_loss / steps

            if epoch == 0:
                best_loss = val_loss
                # save model and parameters
                # self.Generator.save('./weights_best_model.tf',save_format='tf', overwrite=True)
                self.Generator.save_weights(model_path)
            if val_loss < best_loss:
                best_epoch = epoch
                best_loss = val_loss
                # save model and parameters
                # self.Generator.save('./weights_best_model.tf',save_format='tf', overwrite=True)
                self.Generator.save_weights(model_path)
            et = time()
            tt = et - st
            print("Elapsed time: %03d:%02d:%05.2f" % (int(tt / 3600), int(tt / 60) % 60, tt % 60))

        print('val_best_loss:', best_loss, 'best_epoch', best_epoch)


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
        x_data = (x_data-min_x)/(max_x-min_x)

        min_g = np.min(g_data)
        max_g = np.max(g_data)
        g_data = (g_data - min_g) / (max_g - min_g)

        x_data = np.concatenate((x_data, g_data), axis=-1)

        max_t = time_steps - 1
        min_t = 0
        t_data = t_data / max_t
        
        min_y = np.min(y_data)
        max_y = np.max(y_data)
        y_data = (y_data-min_y)/(max_y-min_y)
        normalization_data = [max_x, min_x, max_g, min_g, max_t, min_t, max_y, min_y]
        return x_data, y_data, t_data, normalization_data


if __name__ == '__main__':
    start_time = time() # obtain the starting time of code.

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_probe_x_tiles", type=int, default=48,
                        help=" number of point grid in x axial direction ")
    parser.add_argument("--num_probe_y_tiles", type=int, default=48,
                        help=" number of point grid in y axial direction ")
    parser.add_argument("--time_steps", type=int, default=20,
                        help=" simulation time for each EM trace ")
    parser.add_argument("--percent_valid_split", type=int, default=0.9,
                        help=" 10% of training points is used for validation during training ")
    parser.add_argument("--learning_rate_val", type=int, default=0.0005,
                       help=" learning rate decays exponentially from 0.0005 ")
    parser.add_argument("--num_input_stimuli", type=int, default=750,
                        help=" number of plaintexts ")
    parser.add_argument("--decay_rate_val", type=int, default=0.98,
                        help="  learning rate with the discount factor 0.98 ")
    parser.add_argument("--epochs", type=int, default=100,
                        help="  number of times the model worked on the entire training dataset ")
    parser.add_argument("--batch_size", type=int, default= 64,
                        help="  number of samples processed per training epoch ")
    parser.add_argument("--trained_model_path", type=str, default="./weights/generator_trained_weights.h5",
                        help=" save the trained weight parameters ")
    parser.add_argument("--training_current_map", type=str, default="./current_map_train.npy",
                        help=" input cell current map for model training")
    parser.add_argument("--training_grid_map", type=str, default="./power_grid_map.npy",
                        help=" input power grid map for model training ")
    parser.add_argument("--training_em_map", type=str, default="./em_map_train.npy",
                        help=" input EM map for model training ")

    args = parser.parse_args()

    num_probe_x_tiles = args.num_probe_x_tiles
    num_probe_y_tiles = args.num_probe_y_tiles
    time_steps = args.time_steps
    percent_valid_split = args.percent_valid_split
    learning_rate_val = args.learning_rate_val
    num_input_stimuli = args.num_input_stimuli
    decay_rate_val = args.decay_rate_val
    epochs = args.epochs
    batch_size = args.batch_size
    trained_model_path = args.trained_model_path
    training_current_map = args.training_current_map
    training_grid_map = args.training_grid_map
    training_em_map = args.training_em_map

    print(getToolLogo())
    print('Step 1: Build the cGAN model')
    cgan = CGAN()

    print('Step 2: Load Training Dataset')
    data_train = cgan.read_data(training_current_map, training_grid_map, training_em_map)

    print('Step 3: Pre-process Dataset')
    x_data, y_data, t_data, normalization_data = cgan.process_data(data_train[0], data_train[1], data_train[2], data_train[3])
    
    print('Step 4: GAN model training')

    model = cgan.train_model(x_data, t_data, y_data, trained_model_path)

