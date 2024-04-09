import pickle

with open("./data_buffer_rollout.pkl", "rb") as f:
    data_buffer_rollout = pickle.load(f) #
with open("./data_buffer_rollout_augment.pkl", "rb") as f:
    data_buffer_rollout_augment = pickle.load(f)
with open("./data_buffer_selection.pkl", "rb") as f:
    data_buffer_selection = pickle.load(f)
with open("./data_buffer_selection_augment.pkl", "rb") as f:
    data_buffer_selection_augment = pickle.load(f)

print(data_buffer_rollout)

