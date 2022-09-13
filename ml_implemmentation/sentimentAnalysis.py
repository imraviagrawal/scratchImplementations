"""

You can use any environment and language you are comfortable with. We strongly recommend using a deep learning framework (e.g. PyTorch, Tensorflow) for this interview. Please share your screen as you do so.

If you prefer, you can use a hosted Jupyter notebook which has several deep learning frameworks preinstalled: https://ec2-18-220-48-237.us-east-2.compute.amazonaws.com:8888/ - You will have to click on the screen (there will be no display prompt) and type 'thisisunsafe' to get behind the warning if you are using Chrome, due to a temporarily missing cert. The password is peixuan.

Your task is to build a multiway classifier that will predict the associated sentiment of a given sentence on a 1-5 scale (1=strongly negative, 5=strongly positive).


"""



import requests
import csv
import io
import tensorflow as tf
from sklearn.model_selection import train_test_split

def fetch_data(url):
    """
    Pulls the dataset into a list of lists in the form of:
        [[int: label_1, str: sentence_1],
         [int: label_2, str: sentence_2],
         ...
         [int: label_n, str: sentence_n]]
    """
    r = requests.get(url).content
    raw_data = list(csv.reader(io.StringIO(r.decode('utf-8')), delimiter="\t"))
    data = [[int(label), sent] for label, sent in raw_data]
    return data


class TokenMapper:
    def __init__(self, vocab):
        """
        The straightforward constructor.
        Each token is assigned its own unique int id.

        Params:
            vocab: List[str]
        """
        self.vocab = ['__pab__', "__unk__"] + vocab
        self.vocab_to_id = {token: i for i, token in enumerate(self.vocab)}

    def get_indices(self, sentence):
        """
        Encodes the given sentence into its int representation. This is done on a token level.

        Usage:
        >> mapper.get_indices("This is an example")
        .. [1, 2, 3, 4]

        Params:
            sentence: str

        Return:
            A list of int ids corresponding to each token in `sentence`.
        """
        return [self.vocab_to_id[token] for token in sentence.split(" ")]

    def get_sentence(self, indices):
        """
        Decodes the given indices into its str representation. This is done on a token level.

        Usage:
        >> mapper.get_tokens([1, 2, 3, 4])
        .. "This is an example"

        Params:
            indices: List[str]

        Return:
            The resulting sentence string.
        """
        return " ".join(self.vocab[idx] for idx in indices)

    @classmethod
    def compile(cls, dataset):
        """
        Compiles the given dataset into a TokenMapper.

        Params:
            dataset: List[Tuple[int, str]]

        Return:
            A TokenMapper
        """
        vocab = sorted({token for label, sent in dataset for token in sent.split(" ")})
        return cls(vocab)


# variables
url = "https://raw.githubusercontent.com/pyxyyy/sst-sentiment/master/sst_data.tsv"
batch_size = 64
epochs = 10
lr = 1e-3


data = fetch_data(url)

#tokenzier = TokenMapper()
tokenzier = TokenMapper.compile(data)

# process the dataset
X, y = [tokenzier.get_indices(words) for _, words in data], [label-1 for label, _ in data]

# split the dataset train, validation and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=.50, random_state=42)

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding="post")
# X_test = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding="post")
# X_val = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding="post")


# tensorflow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(500).padded_batch(batch_size)
# val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
# test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# initlize the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # unnormalized output

# model
class Model(tf.keras.Model):
    def __init__(self, vocab_size, emb_dim=64, classes=5):
        super(Model, self).__init__()
        self.emb = tf.keras.layers.Embedding(vocab_size, emb_dim)
        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))
        self.dense1 = tf.keras.layers.Dense(64, activation="relu") # batch*64
        self.dense2 = tf.keras.layers.Dense(classes) # batch * classes

    def call(self, X):
        X_emb = self.emb(X)
        X = self.lstm1(X_emb)
        X = self.lstm2(X)
        X=  self.dense1(X)
        X = self.dense2(X)
        return X

# initialize the model
vocab_len = len(tokenzier.vocab)
model = Model(vocab_len, emb_dim=64, classes=5)

# train function
@tf.function(experimental_relax_shapes=True)
def train_step(X, y):
    with tf.GradientTape() as tape:
        logit = model(X)
        curr_loss = loss(y, logit)

    gradients = tape.gradient(curr_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return tf.argmax(tf.nn.softmax(logit), 1), curr_loss

# training loop

for epoch in range(epochs):
    epoch_avg_loss = tf.keras.metrics.Mean()
    epoch_avg_performance = tf.keras.metrics.Accuracy()

    for i, (X, label) in enumerate(train_dataset.take(500)):
        logit, curr_loss = train_step(X, label)

        epoch_avg_loss.update_state(curr_loss)
        epoch_avg_performance.update_state(label, logit)

        if i % 10:
            print("epoch %s, batch %s, epoch loss %s, epoch Acc %s" %(epoch, i,
                                                                      epoch_avg_loss.result().numpy(), epoch_avg_performance.result().numpy()))
