import tensorflow as tf
import tensorflow_datasets as tfds


# import ipdb; ipdb.set_trace()
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# variables
lr = 1e-3
batchSize = 32
epochs = 30

# tensorflow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(500).batch(batchSize)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batchSize)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(256, activation="relu")
        self.dense3 = tf.keras.layers.Dense(10)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, X, training=False):
        x = self.flatten(X, training=training)
        x = self.dense1(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense2(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense3(x, training=training)
        return x

model = Model()

# training function
@tf.function(experimental_relax_shapes=True)
def train_step(X, y, training=False):
    with tf.GradientTape() as tape:
        logits = model(X, training=training)
        curr_loss = loss(y, logits)

    gradient = tape.gradient(curr_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))

    return tf.argmax(tf.nn.softmax(logits), 1), curr_loss



# training loop
for epoch in range(epochs):
    # initialize metrics
    epoch_avg_loss = tf.keras.metrics.Mean()
    epoch_acc = tf.keras.metrics.Accuracy()

    for i, (X, y) in enumerate(train_dataset.take(500)):
        logits, curr_loss = train_step(X, y, training=True)
        epoch_avg_loss.update_state(curr_loss)
        epoch_acc.update_state(y, logits)

        if i % 50 == 0:
            print("Epoch %s, batch %s, loss %s, acc %s" %(epoch, i, epoch_avg_loss.result().numpy(), epoch_acc.result().numpy()))


