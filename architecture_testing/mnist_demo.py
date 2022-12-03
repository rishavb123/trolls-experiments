import os

import datetime
import tensorflow as tf

from constants import BATCH_SIZE, BUFFER_SIZE, NUM_EPOCHS, RETRAIN_MODEL


def init():
    print(f'TensorFlow version: {tf.__version__}')
    tf.config.run_functions_eagerly(True)



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


class MyModel(tf.keras.models.Model):
    
    def __init__(self):
        super(MyModel, self).__init__()
        self.model_layers = [
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10),
        ]
    
    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x


def main():
    init()

    train_ds, test_ds = get_dataset()
    model = None
    

    if RETRAIN_MODEL:

        model = MyModel()

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                # training=True is only needed if there are layers with different behavior during training versus inference (e.g Dropout)
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, predictions)

        @tf.function
        def test_step(images, labels):
            # training=False is only needed if there are layers with different behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=False)
            t_loss = loss_object(labels, predictions)

            test_loss(t_loss)
            test_accuracy(labels, predictions)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = f'logs/gradient_tape/{current_time}/train'
        test_log_dir = f'logs/gradient_tape/{current_time}/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        print("Starting training . . .")

        for epoch in range(NUM_EPOCHS):
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

            for images, labels in train_ds:
                train_step(images, labels)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

            for test_images, test_labels in test_ds:
                test_step(test_images, test_labels)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

            print(f'Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}, Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result()}')

        model.save('./models/my_model')

    else:

        model = tf.keras.models.load_model('./models/my_model')

    sparse_categorical_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def information_gain_step(images, labels):
        print('---------------------------')

        images = images[0:1]
        labels = labels[0:1]
        print(labels, '-->', end=' ')
        labels = (labels + 5) % 10
        print(labels)

        # training=False is only needed if there are layers with different behavior during training versus inference (e.g. Dropout).
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different behavior during training versus inference (e.g Dropout)
            predictions = model(images, training=False)
            loss = sparse_categorical_loss(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)


        # print(images[0:1])


        print(images.shape, labels.shape)

        info_gain = 0
        for grad in gradients:
            info_gain += tf.math.square(tf.norm(grad))

        print(info_gain)

    for test_images, test_labels in test_ds:
        information_gain_step(test_images, test_labels)

if __name__ == '__main__':
    main()
