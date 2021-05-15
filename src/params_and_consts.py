import os

NUM_EVENTS = 388

vocab = list(range(NUM_EVENTS))
vocab_size = len(vocab)

num_files = 500
num_iters = 200
seq_length = 100

rnn_units = 1024
embedding_dim = 256 
batch_size = 64
learning_rate = 5e-3

start_event = 47

save_path = os.path.join("..", "model.h5")