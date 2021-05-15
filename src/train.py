from model import build_model
from dataset import get_dataset, get_batch
from params_and_consts import *

import tensorflow as tf
import os
import matplotlib.pyplot as plt

def merge_history(old_history, new_history):
    for key in old_history.history.keys():
        old_history.history[key] += new_history.history[key]

    return old_history

def train_model_keras(songs_joined, model):
    for i in range(num_iters):
        x_train, y_train = get_batch(songs_joined, seq_length=seq_length, batch_size=batch_size)
        print(f"\nITERATION NUMBER {i}/{num_iters}\n")
        history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=1)
        
        if i == 0:
            accum_history = history

        merge_history(accum_history, history)

        if i % 100 == 0:
            model.save(save_path)


    model.save(save_path)

    return accum_history

def visualise_training(history):
    plt.plot(history.history['loss'], label='loss')
    plt.title('Training history')
    plt.ylabel('loss')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

def main():
    songs_joined = get_dataset(num_files)
    if os.path.isfile(save_path):
        model = tf.keras.models.load_model(save_path)
    else:
        model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
    history = train_model_keras(songs_joined, model)
    visualise_training(history)

if __name__ == "__main__":
    main()