import tensorflow as tf

class SeqModel(tf.keras.models.Model):

    def __init__(self, layers=[]):
        super(SeqModel, self).__init__()
        self.model_layers = layers

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x

class CNNModel(SeqModel):

    def __init__(self):
        super(CNNModel, self).__init__([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10),
        ])

class CNNModel2(SeqModel):

    def __init__(self):
        super(CNNModel, self).__init__([
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Conv2D(64, 5, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10),
        ])

class DenseModel(SeqModel):
    def __init__(self):
        super(DenseModel, self).__init__([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

ModelClasses = [CNNModel, CNNModel2, DenseModel]