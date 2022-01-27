import tensorflow as tf
import os.path

logs_dir = "/Users/Pablo/Desktop/RL/Logs/"
os.chdir(logs_dir)
os.system("tensorboard --load_fast=false --logdir=logs_dir")