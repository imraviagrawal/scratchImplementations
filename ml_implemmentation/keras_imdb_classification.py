import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import accuracy_score

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)

train_data, test_data = dataset['train'], dataset['test']

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
train_dataset = train_data.shuffle(SHUFFLE_BUFFER_SIZE).padded_batch(BATCH_SIZE)
test_dataset = test_data.padded_batch(BATCH_SIZE)

@tf.function(experimental_relax_shapes=True)
def train_step(X, labels, optimizer, loss):
    with tf.GradientTape() as tape:
        # GET OUTPUT
        logits = model(X)

        # CAL loss
        curr_loss = loss(labels, logits)

    gradients = tape.gradient(curr_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return curr_loss, logits


class Model(tf.keras.Model):
    def __init__(self, vocab, emb_dim):
        super(Model, self).__init__()
        self.emb = tf.keras.layers.Embedding(vocab, emb_dim)
        #self.emb.set_weights(pretrained_weights) if we want to initialize embedding with pretrained weight
        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, input):
        x = self.emb(input) # (batch, seq, emb_dim)
        x = self.lstm1(x)
        x_flat = self.lstm2(x)
        x_dense = self.dense1(x_flat)
        x_op =  self.dense2(x_dense)
        return x_op

# class Model():
#     def __init__(self, word_vocab, loss, optimizer, metrics=["accuracy"]):
#         self.word_vocab = word_vocab
#         self.model = tf.keras.Sequential([
#             tf.keras.layers.Embedding(self.word_vocab, 100),
#             tf.keras.layers.LSTM(64, return_sequences=True),
#             tf.keras.layers.LSTM(64),
#             tf.keras.layers.Dense(64, activation="relu"),
#             tf.keras.layers.Dense(1)
#         ])
#
#         self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
#         self.model.summary()
#
#     def fit(self, train, val, epoch):
#         self.model.fit(train, epochs=epoch)
#
#
#     def predict(self):
#         pass
#
#     def StoreModel(self, filepath):
#         pass
#
#     def loadModel(self, filepath):
#         pass


encoder = info.features['text'].encoder
learning_rate = 1e-3
epochs = 10
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
# model = Model(encoder.vocab_size, loss = loss, optimizer=optimizer)
model = Model(encoder.vocab_size, 64)

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(encoder.vocab_size, 64),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

# model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])



for epoch in range(epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.Accuracy()
    for i, (X, y) in enumerate(train_dataset.take(500)):
        # print(X.shape, y.shape, i)
        curr_loss, logits = train_step(X, y, optimizer, loss)
        epoch_loss_avg.update_state(curr_loss)
        epoch_accuracy.update_state(y, tf.round(tf.nn.sigmoid(logits)))

        if i % 50 == 0:
            # print(accuracy_score(y, tf.round(tf.keras.activations.sigmoid(logits))))
            print(epoch, i, epoch_accuracy.result().numpy(), epoch_loss_avg.result().numpy())


    epoch_accuracy.reset_states()
    epoch_loss_avg.reset_states()


model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30)