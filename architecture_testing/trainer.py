import os

import tensorflow as tf
import datetime

from constants import NUM_EPOCHS

def train_model(model, train_ds, test_ds, loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), log=False):
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    if log:
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
        
        if log:
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        if log:
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

        print(f'Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}, Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result()}')

    model.save(f'./models/{model.__class__.__name__.lower()}')

def make_model(model_cls):
    if os.path.isdir(f'./models/{model_cls.__name__.lower()}'):
        return tf.keras.models.load_model(f'./models/{model_cls.__name__.lower()}'), True
    else:
        return model_cls(), False