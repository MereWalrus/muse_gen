import muspy
import glob
import numpy as np

def midi_dir_to_muspy(path, num_files=float('inf')):
    songs = []

    for i, f in enumerate(glob.iglob(path + '**/**', recursive=True)):
        if f.endswith('.mid'):
            try:
                f_muspy = muspy.inputs.read_midi(f)
                songs.append(f_muspy)
            except:
                pass

            if i >= num_files:
                break

    return songs


def get_dataset(num_files=500):
    songs_muspy = midi_dir_to_muspy("../train_data", num_files)

    for song in songs_muspy:
        song.adjust_resolution(target=muspy.DEFAULT_RESOLUTION)

    songs_repr = []
    for song in songs_muspy:
        songs_repr.append(muspy.to_event_representation(song, encode_velocity=False))

    songs = []
    for song in songs_repr:
        songs.append(np.array([el[0] for el in song]))

    songs_joined = np.concatenate(songs)

    return songs_joined

def get_batch(songs_joined, seq_length, batch_size):
    n = songs_joined.shape[0] - 1
    idx = np.random.choice(n-seq_length, batch_size)

    input_batch = [songs_joined[i : i+seq_length] for i in idx]
    output_batch = [songs_joined[i+1 : i+seq_length+1] for i in idx]

    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])

    return x_batch, y_batch