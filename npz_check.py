import numpy as np


frq_emb = np.load("C://dev//dolphin_initial_testing//DOLPHIN//out_prelbd_task_wise//frequency_emb_stored//3_3//embeddings_samples.npz")
temp_emb= np.load("C://dev//dolphin_initial_testing//DOLPHIN//out_prelbd_task_wise//temporal_emb_stored//3_3//embeddings_samples.npz")
emb = np.load("C://dev//dolphin_initial_testing//DOLPHIN//out_prelbd_task_wise//emb_stored//3_3//embeddings_samples.npz")

print("-----Frequency Embeddings-----")
print("Array 1:", frq_emb['X'])
print("Size of X:", frq_emb['X'].shape)
print("Array 1:", frq_emb['y'])
print("Array 1:", frq_emb['subs'])

print("-----Temporal Embeddings-----")
print("Array 1:", temp_emb['X'])
print("Size of X:", temp_emb['X'].shape)
print("Array 1:", temp_emb['y'])
print("Array 1:", temp_emb['subs'])

print("-----Combined Embeddings-----")
print("Array 1:", emb['X'])
print("Size of X:", emb['X'].shape)
print("Array 1:", emb['y'])
print("Array 1:", emb['subs'])

if emb['X'].all() == emb['X'].all():
    print("The combined embeddings are identical.")
else:
    print("The combined embeddings differ from each other.")