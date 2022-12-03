from constants import BATCH_SIZE, BUFFER_SIZE

import tensorflow as tf

def get_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
    ).batch(BATCH_SIZE)

    return train_ds, test_ds