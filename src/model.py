from params_and_consts import learning_rate
import tensorflow as tf

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(
        rnn_units, 
        return_sequences=True, 
        recurrent_activation='sigmoid',
        stateful=True,
        dropout = 0.4,
    ),
    tf.keras.layers.Dense(vocab_size)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model