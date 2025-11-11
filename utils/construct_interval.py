import numpy as np
import utils.util as util
import solveinequalities.interval as bst_solving_eq


def statistic_condition(eta, a,b,Xtvec):
    sign = np.sign(eta.T.dot(Xtvec)).astype(int)
    u = eta.T.dot(a) * sign * -1
    v = eta.T.dot(b) * sign * -1
    itv = bst_solving_eq.interval_intersection(v,-u)
    return itv
def ReLUcondition(model, a, b, X):
    layers = []
    for name, param in model.named_children():
        temp = dict(param._modules)
        
        for layer_name in temp.values():
            if ('Linear' in str(layer_name)):
                layers.append('Linear')
            elif ('ReLU' in str(layer_name)):
                layers.append('ReLU')
    ptr = 0 

    itv = [(-np.inf, np.inf)]
    weight = None
    bias = None
    for name, param in model.named_parameters():
        if (layers[ptr] == 'Linear'):
            if ('weight' in name):
                weight = np.asarray(param.data.cpu())
            elif ('bias' in name):
                bias = np.asarray(param.data.cpu()).reshape(-1, 1)
                bias = bias.dot(np.ones((1, X.shape[0]))).T
                ptr += 1
                X = X.dot(weight.T) + bias
                a = a.dot(weight.T) + bias
                b = b.dot(weight.T)
        # t2 = time.time()
        if (ptr < len(layers) and layers[ptr] == 'ReLU'):
            ptr += 1

            sign_X = np.sign(X)
            at = (a * -1*sign_X).flatten()
            bt = (b * -1*sign_X).flatten()
            itv = util.interval_intersection(itv, bst_solving_eq.interval_intersection(bt,-at))

            sign_X[sign_X < 0] = 0
            X = X*sign_X
            a = a*sign_X
            b = b*sign_X

            # sub_itv = [(-np.inf, np.inf)]

            # for i in range(X.shape[0]):
            #     for j in range(X.shape[1]):
            #         if X[i][j] > 0:
            #             sub_itv = util.interval_intersection(
            #                 sub_itv, 
            #                 util.solve_quadratic_inequality(a=0, b=-b[i][j], c=-a[i][j])
            #                 )
            #         else:
            #             sub_itv = util.interval_intersection(
            #                 sub_itv, 
            #                 util.solve_quadratic_inequality(a=0, b=b[i][j], c = a[i][j])
            #                 )

            #             X[i][j] = 0
            #             a[i][j] = 0
            #             b[i][j] = 0
            
            # itv = util.interval_intersection(itv, sub_itv)

    return itv, a, b



def KMeancondition(n, K, a, b, initial_centroids, labels_all, members_all,z=0):
    trunc_interval1 = [(-np.inf, np.inf)]
    a = np.asarray(a)
    b = np.asarray(b)

    # Precompute initial centroids' indices for faster access
    initial_centroids_labels = labels_all[0]

    for i in range(n):
        current_label = initial_centroids_labels[i]
        if i == initial_centroids[current_label]:
            continue

        u1 = (a[i] - a[initial_centroids[current_label]]).reshape(-1, 1)
        v1 = (b[i] - b[initial_centroids[current_label]]).reshape(-1, 1)
        p1, q1, o1 = util.construct_p_q_t(u1, v1)

        for k in range(K):
            if k == current_label:
                continue

            u2 = (a[i] - a[initial_centroids[k]]).reshape(-1, 1)
            v2 = (b[i] - b[initial_centroids[k]]).reshape(-1, 1)
            p2, q2, o2 = util.construct_p_q_t(u2, v2)

            p, q, o = (p1 - p2).item(), (q1 - q2).item(), (o1 - o2).item()

            res = util.solve_quadratic_inequality(p, q, o)
            # print("subitv",res)

            trunc_interval1 = util.interval_intersection(trunc_interval1, res)
            # print("Initial K-means truncation interval:", trunc_interval1)
    trunc_interval2 = [(-np.inf, np.inf)]
    for t in range(1, len(labels_all)):
        for i in range(n):
            e_i = np.zeros((1,n))
            e_i[0][i] = 1
            
            gamma_i = np.zeros((1,n))
            label_i = labels_all[t][i] 
            
            C_i_t_minus = list(members_all[t-1][label_i])  
            if len(C_i_t_minus) == 0:
                    continue
    
            gamma_i[:,C_i_t_minus] = 1
        
            E_temp_1 = e_i - gamma_i/len(C_i_t_minus)
            u3 = E_temp_1.dot(a).reshape(-1,1)
            v3 = E_temp_1.dot(b).reshape(-1,1)
            p3, q3, o3 = util.construct_p_q_t(u3, v3)
            for k in range(K):
                e_i = np.zeros((1,n))
                e_i[0][i] = 1
                
                gamma_k = np.zeros((1,n))         
                C_k_t_minus = list(members_all[t-1][k])
                if len(C_k_t_minus) == 0:
                    continue
    
                if k == label_i:
                    continue
                gamma_k[:,C_k_t_minus] = 1
                
            
                E_temp_2 = e_i - gamma_k/len(C_k_t_minus)
                u4 = E_temp_2.dot(a).reshape(-1,1)
                v4 = E_temp_2.dot(b).reshape(-1,1)

                p4, q4, o4 = util.construct_p_q_t(u4, v4)

                p_comma, q_comma, o_comma = (p3 - p4).item(), (q3 - q4).item(), (o3 - o4).item()

                res = util.solve_quadratic_inequality(p_comma, q_comma, o_comma)
                trunc_interval2 = util.interval_intersection(trunc_interval2, res)
    trunc_interval = util.interval_intersection(trunc_interval1, trunc_interval2)
    trunc_interval = [(float(a), float(b)) for (a, b) in trunc_interval]

    return trunc_interval


def KMeancondition2(n, K, A, B, initial_centroids, labels_all, members_all = None, z=0):
    trunc_interval = [(-np.inf, np.inf)]

    # A, B shape n x d
    labels_all = np.asarray(labels_all, dtype=np.int64)  # T x n
    T = labels_all.shape[0]
    labels_for_one_hot = labels_all[:-1, :]

    # build oh as before (ensure labels are ints)
    oh = np.zeros((T-1, K, n), dtype=float)
    rows = np.arange(n)[None, :]
    batches = np.arange(T-1)[:, None]
    oh[batches, labels_for_one_hot, rows] = 1.0

    counts = oh.sum(axis=2)   # shape (T-1, K)
    valid = np.ones((T, K), dtype=bool)
    valid[1:, :] = (counts != 0)

    # normalize safely (we will rely on valid to skip zero-counts)
    sums = oh.sum(axis=2, keepdims=True)
    sums[sums == 0] = 1.0
    oh = oh / sums

    # one_hot and LCA/LCB
    mat = np.zeros((K, n), dtype=float)
    mat[np.arange(K), initial_centroids] = 1.0
    one_hot = np.vstack([mat[None, ...], oh])   # shape (T, K, n)
    LCA = one_hot.dot(A)    # (T, K, d)
    LCB = one_hot.dot(B)

    NCo = np.sum(LCA**2, axis=-1)
    NCq = 2.0 * np.sum(LCA*LCB, axis=-1)
    NCp = np.sum(LCB**2, axis=-1)

    # build self_mask to skip i == centroid (only relevant for t==0)
    self_mask = np.zeros((T, n), dtype=bool)
    init_labels = labels_all[0]
    init_centroid_at_label = np.asarray(initial_centroids, dtype=int)[init_labels]  # length n
    self_mask[0, :] = (np.arange(n) == init_centroid_at_label)

    trunc_interval = [(-np.inf, np.inf)]

    for k in range(K):
        mask_k = (labels_all == k)   # (T, n)
        XA_k = A * mask_k[..., None]  # (T, n, d)
        XB_k = B * mask_k[..., None]

        LCA_k = LCA[:, k, :]   # (T, d)
        LCB_k = LCB[:, k, :]

        for k_ in range(K):
            if k_ == k:
                continue
            LCA_k_ = LCA[:, k_, :]
            LCB_k_ = LCB[:, k_, :]
            LCA_kk = LCA_k - LCA_k_
            LCB_kk = LCB_k - LCB_k_

            XBdotLCB = np.einsum('tnd,td->tn', XB_k, LCB_kk)
            XBdotLCAplusXAdotLCB = (np.einsum('tnd,td->tn', XB_k, LCA_kk) +
                                    np.einsum('tnd,td->tn', XA_k, LCB_kk))
            XAdotLCA = np.einsum('tnd,td->tn', XA_k, LCA_kk)

            P = (NCp[:, k] - NCp[:, k_])[:, None] - 2 * XBdotLCB
            Q = (NCq[:, k] - NCq[:, k_])[:, None] - 2 * XBdotLCAplusXAdotLCB
            O = (NCo[:, k] - NCo[:, k_])[:, None] - 2 * XAdotLCA

            # combine masks: mask_k (points currently labeled k) AND valid for previous cluster k AND not self_mask
            # combined_mask = mask_k & valid[:, k][:, None] & (~self_mask)
            valid_pairs = valid[:, k] & valid[:, k_]  # both clusters non-empty at t>0
            combined_mask = (mask_k & valid_pairs[:, None] & (~self_mask))
            P_sel = P[combined_mask]
            Q_sel = Q[combined_mask]
            O_sel = O[combined_mask]

            if P_sel.size == 0:
                continue
            trunc_interval = util.interval_intersection(trunc_interval,solveinterval(P_sel, Q_sel, O_sel, z))

    trunc_interval = [(float(a), float(b)) for (a, b) in trunc_interval]
    return trunc_interval


def KMeancondition3(n, K, A, B, initial_centroids, labels_all, members_all, z=0):
    trunc_interval = [(-np.inf, np.inf)]
    labels_all = np.asarray(labels_all)  # Shape: T x n
    T = labels_all.shape[0]

    # One-hot encoding for labels (T-1 x K x n)
    labels_for_one_hot = labels_all[:-1]
    oh = np.zeros((T-1, K, n))
    oh[np.arange(T-1)[:, None], labels_for_one_hot, np.arange(n)] = 1.0
    sums = oh.sum(axis=2, keepdims=True)
    sums[sums == 0] = 1.0
    oh /= sums

    # Initial centroids one-hot (1 x K x n)
    mat = np.zeros((K, n))
    mat[np.arange(K), initial_centroids] = 1.0
    one_hot = np.concatenate([mat[None], oh], axis=0)

    # Compute LCA, LCB (T x K x d)
    LCA = one_hot @ A
    LCB = one_hot @ B

    # Compute NCo, NCq, NCp (T x K)
    NCo = np.sum(LCA**2, axis=-1)
    NCq = 2.0 * np.sum(LCA * LCB, axis=-1)
    NCp = np.sum(LCB**2, axis=-1)

    # Mask and XA, XB
    mask = (labels_all[..., None] == np.arange(K))  # T x n x K
    XA = A[None, :, None, :] * mask[..., None]  # T x n x K x d
    XB = B[None, :, None, :] * mask[..., None]  # T x n x K x d

    # Differences
    LCA_diff = LCA[:, :, None, :] - LCA[:, None, :, :]  # T x K x K x d
    LCB_diff = LCB[:, :, None, :] - LCB[:, None, :, :]  # T x K x K x d

    # Dot products
    XBdotLCB = np.einsum('tnkd,tkjd->tnkj', XB, LCB_diff)  # T x n x K x K
    XBdotLCA = np.einsum('tnkd,tkjd->tnkj', XB, LCA_diff)
    XAdotLCB = np.einsum('tnkd,tkjd->tnkj', XA, LCB_diff)
    XBdotLCA_XAdotLCB = XBdotLCA + XAdotLCB
    XAdotLCA = np.einsum('tnkd,tkjd->tnkj', XA, LCA_diff)

    # P, Q, O
    diff_NCp = NCp[:, :, None] - NCp[:, None, :]  # T x K x K
    P = diff_NCp[:, None, :, :] - 2 * XBdotLCB  # T x n x K x K

    diff_NCq = NCq[:, :, None] - NCq[:, None, :]
    Q = diff_NCq[:, None, :, :] - 2 * XBdotLCA_XAdotLCB

    diff_NCo = NCo[:, :, None] - NCo[:, None, :]
    O = diff_NCo[:, None, :, :] - 2 * XAdotLCA

    # Valid mask for k != j
    eye = np.eye(K)[None, None, :, :]
    mask_valid = mask[:, :, :, None] * (1 - eye)  # T x n x K x K

    # Flatten valid entries
    indices = np.nonzero(mask_valid)
    P_flat = P[indices]
    Q_flat = Q[indices]
    O_flat = O[indices]

    # Intersect intervals
    new_intervals = solveinterval(P_flat, Q_flat, O_flat, z)
    trunc_interval = util.interval_intersection(trunc_interval, new_intervals)

    return [(float(a), float(b)) for (a, b) in trunc_interval]
def solveinterval(P: np.ndarray, Q: np.ndarray, O: np.ndarray, z):
    """
    P, Q, O: 1-D NumPy arrays of the same length.
    Returns: list of (low, high) tuples after intersecting intervals.
    """
    trunc_interval = [(-np.inf, np.inf)]
    
    # vectorized iteration with NumPy
    for p, q, o in zip(P, Q, O):
        # print("tensor pqo", f"{p:.5f}", f"{q:.5f}", f"{o:.5f}")
        # print("pzz qz o", f"{p*z*z + q*z + o:.5f}")
        res = util.solve_quadratic_inequality(p, q, o)
        trunc_interval = util.interval_intersection(trunc_interval, res)
    # print("----------------- interval",trunc_interval)
    return trunc_interval

# import cupy as cp

# def KMeanconditionCUPY(n, K, A, B, initial_centroids, labels_all, members_all, z=0):
#     trunc_interval = [(-np.inf, np.inf)]

#     # Move arrays to GPU in float64
#     A = cp.asarray(A, dtype=cp.float64)
#     B = cp.asarray(B, dtype=cp.float64)
#     labels_all = cp.asarray(labels_all)  # keep int type for indexing

#     T = labels_all.shape[0]

#     # One-hot encoding for labels (T-1 x K x n)
#     labels_for_one_hot = labels_all[:-1]
#     oh = cp.zeros((T-1, K, n), dtype=cp.float64)
#     oh[cp.arange(T-1)[:, None], labels_for_one_hot, cp.arange(n)] = 1.0
#     sums = oh.sum(axis=2, keepdims=True)
#     sums = cp.where(sums == 0, 1.0, sums)
#     oh /= sums

#     # Initial centroids one-hot (1 x K x n)
#     mat = cp.zeros((K, n), dtype=cp.float64)
#     mat[cp.arange(K), initial_centroids] = 1.0
#     one_hot = cp.concatenate([mat[None], oh], axis=0)

#     # Compute LCA, LCB (T x K x d)
#     LCA = one_hot @ A
#     LCB = one_hot @ B

#     # Compute NCo, NCq, NCp (T x K)
#     NCo = cp.sum(LCA**2, axis=-1)
#     NCq = 2.0 * cp.sum(LCA * LCB, axis=-1)
#     NCp = cp.sum(LCB**2, axis=-1)

#     # Mask and XA, XB
#     mask = (labels_all[..., None] == cp.arange(K))  # T x n x K
#     XA = A[None, :, None, :] * mask[..., None]  # T x n x K x d
#     XB = B[None, :, None, :] * mask[..., None]  # T x n x K x d

#     # Differences
#     LCA_diff = LCA[:, :, None, :] - LCA[:, None, :, :]  # T x K x K x d
#     LCB_diff = LCB[:, :, None, :] - LCB[:, None, :, :]  # T x K x K x d

#     # Dot products
#     XBdotLCB = cp.einsum('tnkd,tkjd->tnkj', XB, LCB_diff)
#     XBdotLCA = cp.einsum('tnkd,tkjd->tnkj', XB, LCA_diff)
#     XAdotLCB = cp.einsum('tnkd,tkjd->tnkj', XA, LCB_diff)
#     XBdotLCA_XAdotLCB = XBdotLCA + XAdotLCB
#     XAdotLCA = cp.einsum('tnkd,tkjd->tnkj', XA, LCA_diff)

#     # P, Q, O
#     diff_NCp = NCp[:, :, None] - NCp[:, None, :]
#     P = diff_NCp[:, None, :, :] - 2 * XBdotLCB

#     diff_NCq = NCq[:, :, None] - NCq[:, None, :]
#     Q = diff_NCq[:, None, :, :] - 2 * XBdotLCA_XAdotLCB

#     diff_NCo = NCo[:, :, None] - NCo[:, None, :]
#     O = diff_NCo[:, None, :, :] - 2 * XAdotLCA

#     # Valid mask for k != j
#     eye = cp.eye(K, dtype=cp.float64)[None, None, :, :]
#     mask_valid = mask[:, :, :, None] * (1 - eye)

#     # Flatten valid entries
#     indices = cp.nonzero(mask_valid)
#     P_flat = P[indices]
#     Q_flat = Q[indices]
#     O_flat = O[indices]

#     # >>>> MOVE TO CPU for solveinterval
#     P_flat_cpu = cp.asnumpy(P_flat)
#     Q_flat_cpu = cp.asnumpy(Q_flat)
#     O_flat_cpu = cp.asnumpy(O_flat)

#     new_intervals = solveinterval(P_flat_cpu, Q_flat_cpu, O_flat_cpu, z)

#     # Assuming util.interval_intersection works with NumPy
#     trunc_interval = util.interval_intersection(trunc_interval, new_intervals)

#     return [(float(a), float(b)) for (a, b) in trunc_interval]


