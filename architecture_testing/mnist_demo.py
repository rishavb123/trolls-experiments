import os

import datetime
import tensorflow as tf

from constants import BATCH_SIZE, BUFFER_SIZE, NUM_EPOCHS, RETRAIN_MODEL
from data import get_dataset
from model_structures import CNNModel
from trainer import train_model, make_model
from information_gain import calculate_information_gains


def init():
    print(f'TensorFlow version: {tf.__version__}')
    tf.config.run_functions_eagerly(True)


MyModel = CNNModel

def main():
    init()

    train_ds, test_ds = get_dataset()
    model, trained = make_model(MyModel)

    if RETRAIN_MODEL or not trained:
        train_model(model, train_ds, test_ds)

    calculate_information_gains(model, test_ds)

if __name__ == '__main__':
    main()
