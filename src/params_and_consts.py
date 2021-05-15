import os

NUM_EVENTS = 388

vocab = list(range(NUM_EVENTS))
vocab_size = len(vocab)

num_files = 1000
num_iters = 5000
seq_length = 100

rnn_units = 1024
embedding_dim = 256 
batch_size = 64
learning_rate = 5e-4

start_event = 47
generation_length = 100

save_path = os.path.join("..", "model.h5")