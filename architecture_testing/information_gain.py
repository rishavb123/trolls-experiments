import tensorflow as tf

# from constants import ALPHA, BETA

def calculate_information_gains(model, test_ds, loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)):
    @tf.function
    def information_gain_step(images, labels, idx=0, label_offset=1):
        print('---------------------------')

        images = images[idx: idx + 1]
        labels = labels[idx: idx + 1]
        print(tf.get_static_value(labels), '-->', end=' ')
        labels = (labels + label_offset) % 10
        print(tf.get_static_value(labels))

        with tf.GradientTape(persistent=True) as tape:
            predictions = model(images, training=False)
            loss = loss_object(labels, predictions)

        def calc_gradients(vars):
            gradients = tape.gradient(loss, vars)
            norm_grads = 0
            for grad in gradients:
                norm_grads += tf.math.square(tf.norm(grad))
            return norm_grads

        theta_grad_norm = calc_gradients(model.trainable_variables)

        # del tape

        info_gain = tf.get_static_value(theta_grad_norm)
        # info_gain = ALPHA * theta_grad_norm + BETA * inpt_grad_norm

        print(info_gain)

        return info_gain

    c = 0

    for test_images, test_labels in test_ds:
        c += 1
        if c > 10: break
        information_gain_step(test_images, test_labels)