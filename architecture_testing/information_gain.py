import numpy as np
import tensorflow as tf
from tqdm import tqdm

from constants import ALPHA, BETA


def calculate_information_gains(model, test_ds, loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), log=False):
    lprint = lambda *args, **kwargs: print(*args, **kwargs) if log else None

    @tf.function
    def information_gain_step(images, labels, idx=0, set_label=None, label_offset=1):

        lprint('---------------------------')


        images = images[idx: idx + 1]
        labels = labels[idx: idx + 1]

        old_label = tf.get_static_value(labels)[0]
        lprint(old_label, '-->', end=' ')


        labels = (labels + label_offset) % 10
        if set_label is not None:
            labels = tf.constant([set_label]) 

        new_label = tf.get_static_value(labels)[0]
        lprint(new_label)

        images = [tf.Variable(inp) for inp in images]

        with tf.GradientTape(persistent=True) as tape:
            predictions = model(images, training=False)
            loss = loss_object(labels, predictions)

        def calc_gradients(vars):
            gradients = tape.gradient(loss, vars)
            norm_grads = 0
            for grad in gradients:
                norm_grads += tf.math.square(tf.norm(grad))
            return norm_grads

        theta_grad_norm = 0
        inpt_grad_norm = 0

        if ALPHA > 0:
            theta_grad_norm = calc_gradients(model.trainable_variables)
        if BETA > 0:
            inpt_grad_norm = calc_gradients(images)

        theta_grad_norm = tf.get_static_value(theta_grad_norm)
        inpt_grad_norm = tf.get_static_value(inpt_grad_norm)

        info_gain = ALPHA * theta_grad_norm + BETA * inpt_grad_norm

        lprint(info_gain)

        del tape

        return info_gain, old_label, new_label, theta_grad_norm, inpt_grad_norm


    total_info_gains = np.zeros((10, 10))
    total_theta_info_gains = np.zeros((10, 10))
    total_inpt_info_gains = np.zeros((10, 10))
    counts = np.zeros((10, 10))

    for test_images, test_labels in tqdm(test_ds):
        n = len(test_images)
        for new_label in range(10):
            for i in range(n):
                info_gain, old_label, new_label, theta_info_gain, inpt_info_gain = information_gain_step(test_images, test_labels, idx=i, set_label=new_label)
                total_info_gains[old_label, new_label] += info_gain
                total_theta_info_gains += theta_info_gain
                total_inpt_info_gains += inpt_info_gain
                counts[old_label, new_label] += 1

    return total_info_gains / counts, total_theta_info_gains / counts, total_inpt_info_gains / counts