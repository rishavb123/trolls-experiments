import tensorflow as tf

def calculate_information_gains(model, test_ds, loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)):
    @tf.function
    def information_gain_step(images, labels):
        print('---------------------------')

        images = images[0:1]
        labels = labels[0:1]
        print(labels, '-->', end=' ')
        labels = (labels + 5) % 10
        print(labels)

        with tf.GradientTape() as tape:
            predictions = model(images, training=False)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)


        # print(images[0:1])


        print(images.shape, labels.shape)

        info_gain = 0
        for grad in gradients:
            info_gain += tf.math.square(tf.norm(grad))

        print(info_gain)

    for test_images, test_labels in test_ds:
        information_gain_step(test_images, test_labels)