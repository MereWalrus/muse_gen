import matplotlib.pyplot as plt

# 0-127 to encode note-one events, 128-255 for note-off events, 256-355 for time-shift events, and 356 to 387 for velocity
def arr_to_event_types(arr):
    types = []
    for num in arr:
        if num >= 0 and num <= 127:
            types.append(f"note-on: {num}")
        elif num >= 128 and num <= 255:
            types.append(f"note-off: {num - 128}")
        elif num >= 256 and num <= 355:
            types.append(f"time-shift: {num - 255}")
        elif num >= 356 and num <= 387:
            types.append("velocity")
        else:
            types.append("unknown")

    return types

def plot_event_distrib(events, num_events):
  plt.clf()
  num_bins = num_events
  n, bins, patches = plt.hist(events, num_bins, facecolor='red')