import numpy as np
import pathlib as Path

def emb_sanity_check():

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
#-----------------------------    
# Embedding consistency check
#-----------------------------

def check_embedding_consistency(emb_1_path , emb_2_path):
    emb_1 = np.load(emb_1_path)
    emb_2 = np.load(emb_2_path)    

    if emb_1['X'].all() == emb_2['X'].all():
        print("The combined embeddings are identical.")
    else:
        print("The combined embeddings differ from each other.")


def embeding_chceck_iterator(root_path1 , root_path2):

    emb1_root = Path.Path(root_path1)
    emb2_root = Path.Path(root_path2)
    
    emb1 = emb1_root.rglob("embeddings_samples.npz")
    emb2 = emb2_root.rglob("embeddings_samples.npz")
    
    for e1, e2 in zip(emb1, emb2):
        print(f"Checking embeddings: {e1.name} and {e2.name}")
        check_embedding_consistency(e1, e2)
def main():
    # Single check
    # emb_sanity_check()

    # Iterative check
    root_path1 = "C://dev//dolphin_initial_testing//DOLPHIN//out_prelbd_task_wise//emb_stored"
    root_path2 = "C://dev//dolphin_initial_testing//DOLPHIN//out_prelbd_task_wise//emb_stored_v2"
    embeding_chceck_iterator(root_path1 , root_path2)

if __name__ == "__main__":
    main()

