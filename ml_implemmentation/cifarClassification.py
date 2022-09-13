import tensorflow as tf
import tensorflow_datasets as tfds


# import ipdb; ipdb.set_trace()
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

lr = 1e-3
epochs = 10
batch_size = 32
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(100).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.maxPool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10)
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.maxPool1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


model = Model()

@tf.function(experimental_relax_shapes=True)
def train_step(X, y):
    with tf.GradientTape() as tape:
        logits = model(X)
        curr_loss = loss(y, logits)

    # cal gradeints
    gradients = tape.gradient(curr_loss, model.trainable_variables)

    # apply gradeints
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return tf.argmax(tf.nn.softmax(logits), 1), curr_loss


# training loop
for epoch in range(epochs):
    # initialize metrics
    # mean loss
    epoch_acc_score = tf.keras.metrics.Accuracy()
    epoch_avg_loss = tf.keras.metrics.Mean()

    for i, (X, y) in enumerate(train_dataset.take(500)):
        logits, curr_loss = train_step(X, y)

        epoch_avg_loss.update_state(curr_loss)
        epoch_acc_score.update_state(y, logits)
        # import ipdb; ipdb.set_trace()

        if i % 50 == 0:
            print(epoch, i, epoch_avg_loss.result().numpy(), epoch_acc_score.result().numpy())