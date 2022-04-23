#!/usr/bin/env python3
import argparse
from scripts.training import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Runs the experiment with necessary parameters',
            epilog='Leave space after each parameter, eg --dir logs',
            allow_abbrev=False)
    parser.add_argument('--image_dir',
            nargs='?', default='images/temp/',
            help='Image directory to store experiment images'
                 '(default: %(default)s)')
    parser.add_argument('--timesteps',
            nargs='?', default=1e6, type=float,
            help='Number of timesteps to train'
                 '(default: %(default)s)')
    parser.add_argument('--batch_size',
            nargs='?', default=64, type=int,
            help='Batch size for training'
                 '(default: %(default)s)')
    parser.add_argument('--buffer_size',
            nargs='?', default=50000, type=int,
            help='Buffer size for past observations'
                 '(default: %(default)s)')
    parser.add_argument('--model_dir',
            nargs='?', default='train/temp/', type=str,
            help='Directory for saving models'
                 '(default: %(default)s)')
    dct = vars(parser.parse_args())
    print(dct)
