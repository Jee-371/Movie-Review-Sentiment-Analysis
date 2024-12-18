!wget https://www.dropbox.com/s/pdhwlpi2yeie0ol/movie-reviews-dataset.zip
!unzip -q "/content/movie-reviews-dataset.zip"

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import text_dataset_from_directory
from tensorflow.strings import regex_replace
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, RNN, SimpleRNNCell, Embedding, LSTM, GRU, Bidirectional

def prepare_data(dir):
    data = text_dataset_from_directory(dir)
    return data.map(lambda text, label: (regex_replace(text, '<br />', ' '), label))

train_data = prepare_data('movie-reviews-dataset/train')
test_data = prepare_data('movie-reviews-dataset/test')

for text_batch, label_batch in train_data.take(1):
    print(text_batch.numpy()[0])
    print(label_batch.numpy()[0])

max_tokens = 1000
max_len = 100

vectorize_layer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_len,
)

train_texts = train_data.map(lambda text, label: text)
vectorize_layer.adapt(train_texts)

def build_model(rnn_layer):
    model = Sequential([
        Input(shape=(1,), dtype="string"),
        vectorize_layer,
        Embedding(max_tokens + 1, 128),
        rnn_layer,
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])
    return model

rnn_layers = [
    RNN(SimpleRNNCell(64), return_sequences=False),
    LSTM(64),
    GRU(64),
    Bidirectional(LSTM(64)),
    Bidirectional(GRU(64))
]

histories = []
models = []
for rnn_layer in rnn_layers:
    model = build_model(rnn_layer)
    history = model.fit(train_data, epochs=20)
    histories.append(history)
    models.append(model)

for model in models:
    model.evaluate(test_data)

text = """
Just left the theatre after watching the film, and the effect the film had on me is still there...
I am still in that zone and I don't want to withdraw the emotions I felt in the theatre...
true to the film's name it justifies the way love was portrayed in the film in a pure and innocent
way and also the depth in all the relationships in the film..there are so many feelings and emotions
in one film and no one else could have done it except Mamooka ....all the cast members were perfect for the roles...
hats off to the story, direction, music, and everything and everybody involved behind the screen...
"""

for model in models:
    print(model.predict([text]))

plt.figure(figsize=(10, 6))
for i, history in enumerate(histories):
    plt.plot(history.history['accuracy'], label=f'Model {i + 1}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

for i, history in enumerate(histories):
    accuracy = history.history['accuracy'][-1]
    print(f"Accuracy of Model {i + 1}: {accuracy * 100:.2f}%")
