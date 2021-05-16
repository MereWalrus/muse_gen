from params_and_consts import save_path, vocab_size, embedding_dim, rnn_units, start_event, generation_length
from model import build_model

import muspy
import tensorflow as tf
import numpy as np
import os

def get_gen_model():
    gen_model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

    trained_model = tf.keras.models.load_model(save_path)
    old_weights = trained_model.get_weights()
    gen_model.set_weights(old_weights)

    return gen_model

def generate_notes(model, start, generation_length=100):
    input_seq = start
    input_seq = tf.expand_dims(input_seq, 0)

    notes_generated = []

    model.reset_states()

    for _ in range(generation_length):
        predictions = model(input_seq)
        predictions = tf.squeeze(predictions, 0)
            
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        notes_generated.append(predicted_id)
        predicted_id_exp = tf.constant([predicted_id], dtype=np.int32)

        input_seq = tf.concat([input_seq[0], predicted_id_exp], 0)
        input_seq = tf.expand_dims(input_seq, axis=0)

    return (start + notes_generated)


def arr_to_event_repr(arr):
    return np.array([[el] for el in arr])

def main():
    gen_model = get_gen_model()
    generated_notes = generate_notes(gen_model, start=[start_event], generation_length=generation_length)
    generated_events = arr_to_event_repr(generated_notes)
    generated_muspy = muspy.inputs.from_event_representation(generated_events)
    midi_program = 0
    generated_muspy[0].program = midi_program
    #midi_program = songs_muspy[0][0].program
    # generated_muspy[0].is_drum = True

    gen_midi_path = os.path.join("..", "gen.mid")
    muspy.write_midi(gen_midi_path, generated_muspy)
    # muspy.visualization.show_pianoroll(generated_muspy)

if __name__ == "__main__":
    main()