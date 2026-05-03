import numpy as np

def check_observability(H):

    rank = np.linalg.matrix_rank(H)

    if rank < H.shape[1]:
        print("❌ System NOT observable")
    else:
        print("✅ System observable")