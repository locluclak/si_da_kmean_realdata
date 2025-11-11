import numpy as np
from scipy import stats
import torch
import matplotlib.pyplot as plt
import yaml
import time
from tqdm import tqdm
import random
from gpu_accelerate import operations, conditioning
from sklearn.metrics import adjusted_rand_score

import utils.construct_interval as construct_interval
from utils.kmeans import kmeans
import utils.util as util
from models.wdgrl import WDGRL

device = "cpu"
# def conditional_power(M, )

def test_statistic(X_vec, Xt, ns, nt, d, n_clusters, Sigma, labels_all_obs,return_sign=False, pair=None):
    if pair is not None:
        c1, c2 = pair
    else:
        c1, c2 = np.random.choice(n_clusters, 2, replace=False)
    idx_cluster_c1 = np.argwhere(labels_all_obs[-1][ns:] == c1).flatten()
    idx_cluster_c2 = np.argwhere(labels_all_obs[-1][ns:] == c2).flatten()
    if idx_cluster_c1.size == 0 or idx_cluster_c2.size == 0:
        return None

    I_d = np.identity(d)
    
    eta_c1_idx = np.zeros((nt, 1))
    eta_c1_idx[idx_cluster_c1] = 1 / len(idx_cluster_c1)
    eta_c1 = np.kron(I_d, eta_c1_idx)
    
    
    eta_c2_idx = np.zeros((nt, 1))
    eta_c2_idx[idx_cluster_c2] = 1 / len(idx_cluster_c2)
    eta_c2 = np.kron(I_d, eta_c2_idx)
   
    eta_tmp = eta_c1 - eta_c2
    sign_tmp = np.dot(eta_tmp.T, vec(Xt))
    sign = np.sign(sign_tmp).astype(int)
    # print("eta_tmp", eta_tmp)
    if return_sign:
        return sign
    eta_sign = np.dot(eta_tmp, sign)

    eta = np.vstack((np.zeros((ns*d, 1)), eta_sign))
    etaTXvec = np.dot(eta.T, X_vec)

    etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta)
    b = np.dot(np.dot(Sigma, eta), np.linalg.inv(etaT_Sigma_eta))
    a = np.dot(np.identity(X_vec.shape[0]) - np.dot(b, eta.T), X_vec)
    z = etaTXvec.item()
    
    return {
        "a": a,
        "b": b,
        "eta_tmp": eta_tmp,
        "zobs": z,
        "etaT_Sigma_eta": etaT_Sigma_eta.item(),
        "c1": c1,
        "c2": c2,
        "cluster_c1_obs": idx_cluster_c1,
        "cluster_c2_obs": idx_cluster_c2,
        "sign": sign
    }

def overconditioning(model, X,ns , eta, a, b, np_wdgrl, n_clusters, initial_centroids_obs, labels_all_obs,z=0,X_=None):
    interval_da, a_, b_ = construct_interval.ReLUcondition(model.encoder, a, b, X)

    interval_kmean = construct_interval.KMeancondition2(X.shape[0], n_clusters, a_, b_, initial_centroids_obs, labels_all_obs,z)
    # interval_kmean = construct_interval.KMeancondition(X.shape[0], n_clusters, a_, b_, initial_centroids_obs, labels_all_obs, members_all_obs,z)
    # interval_kmean = construct_interval.KMeanconditionCUPY(X.shape[0], n_clusters, a_, b_, initial_centroids_obs, labels_all_obs, members_all_obs,z)
    interval_test_statistic = construct_interval.statistic_condition(eta, vec(a[ns:]), vec(b[ns:]), vec(X[ns:]))

    final_interval = util.interval_intersection(interval_test_statistic,
                      util.interval_intersection(interval_da, interval_kmean))
    return final_interval
def parametric(model, X,ns, a, b, eta, np_wdgrl, n_clusters, c1, c2, c1_obs, c2_obs, signobs, zmin = -20, zmax = 20, log=None, seed=None,device="cpu"):
    n, d = X.shape
    z =  zmin
    zmax = zmax
    countitv=0
    Z = []
    stepsize= 0.00001

    total_steps = int((zmax - zmin) / stepsize)
    with tqdm(total=total_steps, desc=f"Seed {seed}") as pbar:
        while z < zmax:
            z += stepsize
            # print("z =",z)
            Xdeltaz = a + b*z
            Xdeltaz_torch = torch.from_numpy(Xdeltaz).double().to(device)
            with torch.no_grad():
                # Xdeltaz_transformed = final_model.extract_feature(Xdeltaz_torch).cpu().numpy()
                Xdeltaz_transformed = model.extract_feature(Xdeltaz_torch).cpu().numpy()
            initial_centroids_z, labels_all_z, members_all_obs = kmeans(Xdeltaz_transformed, n_clusters)
            
            # print("sum xt",np.sum(Xdeltaz[ns:]))
            # sign_z = test_statistic(vec(Xdeltaz), Xdeltaz[ns:], ns, nt, d, n_clusters, Sigma=None, labels_all_obs=labels_all_z,return_sign=True)
            sign_z = np.sign(eta.T.dot(vec(Xdeltaz[ns:])))
            oc = overconditioning(model, Xdeltaz,ns, eta, a, b, np_wdgrl, n_clusters, initial_centroids_z, labels_all_z, z=z,X_=Xdeltaz_transformed)
            idx_cluster_c1 = np.argwhere(labels_all_z[-1][ns:] == c1).flatten()
            idx_cluster_c2 = np.argwhere(labels_all_z[-1][ns:] == c2).flatten()

            if np.array_equal(c1_obs, idx_cluster_c1) and np.array_equal(c2_obs, idx_cluster_c2) and np.array_equal(signobs, sign_z):
                Z = util.interval_union(Z, oc)
                countitv+=1
            # if 1:
            #     print("sign obs",signobs.reshape(1,-1))
            #     print("sign z  ",sign_z.reshape(1,-1))
            # print("all oc:", oc)
            # print("z :", z)
            z = oc[-1][1] # ruv
            # en = time.time()
            # with open(f"./experiments/time_{n}_{p}.txt", "a") as f:
            #     f.write(f"{en-st}\n")
            pbar.update(int((z - zmin) / stepsize) - pbar.n)

    if log is not None:
        with open(log, "a") as f:
            f.write(f"Number of intervals: {countitv}\n\n")
            f.write(f"Final interval: {Z}\n")
    return Z

def test_statistic_permutationtest(Xt, idx_cluster_c1, idx_cluster_c2,ns,nt,d):
    I_d = np.identity(d)
    
    eta_c1_idx = np.zeros((nt, 1))
    eta_c1_idx[idx_cluster_c1] = 1 / len(idx_cluster_c1)
    eta_c1 = np.kron(I_d, eta_c1_idx)
    
    
    eta_c2_idx = np.zeros((nt, 1))
    eta_c2_idx[idx_cluster_c2] = 1 / len(idx_cluster_c2)
    eta_c2 = np.kron(I_d, eta_c2_idx)
   
    eta_tmp = eta_c1 - eta_c2
    etaTx = np.abs(np.dot(eta_tmp.T, vec(Xt)))
    # print(np.sum(etaTx))

    return np.sum(etaTx)


def permutation_test(Xt, idx_cluster_c1, idx_cluster_c2, 
                     test_statistic_func, ns, nt, d, n_permutations=1000, random_state=None):
    """
    Permutation test for checking if two clusters differ significantly.
        
    Returns
    -------
    observed_stat : float
        Test statistic on the real data.
    p_value : float
        p-value from permutation test.
    permuted_stats : np.ndarray
        Distribution of permuted test statistics.
    """
    rng = np.random.default_rng(random_state)

    # observed test statistic
    observed_stat = test_statistic_func(Xt, idx_cluster_c1, idx_cluster_c2,ns,nt,d)

    # combine indices and labels
    all_indices = np.concatenate([idx_cluster_c1, idx_cluster_c2])
    n1 = len(idx_cluster_c1)

    permuted_stats = np.zeros(n_permutations)

    for i in range(n_permutations):
        # shuffle all indices
        rng.shuffle(all_indices)
        perm_idx1 = all_indices[:n1]
        perm_idx2 = all_indices[n1:]
        
        permuted_stats[i] = test_statistic_func(Xt, perm_idx1, perm_idx2,ns,nt,d)

    # two-sided p-value
    p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))

    return {
        "observed_stat": observed_stat,
        "p_value": p_value,
        "permuted_stats": permuted_stats
    }

def vec(A):
    vec = A.reshape(-1)
    return vec.reshape(-1,1)


def parametric_test(final_model, Xs, Xt, Sigma, K, device, _=None):

    # global ns, nt, d

    ns = Xs.shape[0]
    nt = Xt.shape[0]

    d = Xs.shape[1]
    n = ns + nt

    Xs_torch = torch.from_numpy(Xs).double().to(device)
    Xt_torch = torch.from_numpy(Xt).double().to(device)

    with torch.no_grad():
        xs_hat = final_model.extract_feature(Xs_torch).cpu().numpy()
        xt_hat = final_model.extract_feature(Xt_torch).cpu().numpy()
    
    Xs_vec = vec(Xs)
    Xt_vec = vec(Xt)
    X_vec = np.vstack((Xs_vec, Xt_vec))
    X_origin = np.vstack((Xs, Xt))
    X_transformed = np.vstack((xs_hat, xt_hat))

    initial_centroids_obs, labels_all_obs, members_all_obs = kmeans(X_transformed, K)

    # labelkmean = labels_all_obs[-1][ns:]
    # print("ari",adjusted_rand_score(Yt, labelkmean))
    # Sigma = np.identity(n*d)
    try:
        a, b, eta_tmp, etaTX, etaT_Sigma_eta, c1, c2, c1_obs, c2_obs, sign = test_statistic(X_vec, Xt, ns, nt, d, K, Sigma, labels_all_obs).values()
        
    except Exception as e:
        print("test statistic is none", e) 
        return None
    # print(mut.shape)
    # print(mut[c1_obs].shape)
    # print(mut[c2_obs].shape)

    # check_h1 =np.sum(np.abs(np.mean(mut[c1_obs], axis=0) - np.mean(mut[c2_obs], axis=0))) 
    # # print(check_h1)
    # if abs(check_h1) < 0.01:
    #     print("Incorrect cluster", check_h1)
    #     return None
    # for k in [c1,c2]:
    #     cluster_mask = (labelkmean == k).astype(int)
    #     true_mask = (Yt == np.bincount(Yt[labelkmean == k]).argmax()).astype(int)
    #     ari = adjusted_rand_score(true_mask, cluster_mask)
    #     if ari <0.95:
    #         return None
    # permutation_test_pvalue = permutation_test(Xt, c1_obs, c2_obs, test_statistic_permutationtest,)["p_value"]
    # print(permutation_test_pvalue)
    # with open(f'logs/selective_inference_log/TPRpermutation_p_valueslist_delta{delta}.txt', 'a') as f:
    #     f.write(f"{permutation_test_pvalue}\n")
    # return permutation_test_pvalue
    a_2d = a.reshape(n, d)
    b_2d = b.reshape(n, d)

    np_wdgrl = None# operations.convert_network_to_numpy(final_model.encoder)
    # final_model.encoder = final_model.encoder.to(device)


    std = np.sqrt(etaT_Sigma_eta)


    # final_interval = overconditioning(final_model, X_origin,eta_tmp, a_2d, b_2d,np_wdgrl, K, initial_centroids_obs, labels_all_obs, members_all_obs,z=etaTX, X_=X_transformed)
    final_interval = parametric(final_model, 
                                X_origin, ns, 
                                a_2d, 
                                b_2d,
                                eta_tmp,
                                np_wdgrl, 
                                K, c1, c2, c1_obs, c2_obs, 
                                signobs = sign, 
                                zmin=-20*std, zmax=20*std,
                                )
    # final_interval = [(-np.inf, np.inf)]
    
    # print(etaTX)
    # print("Final interval",final_interval)
    selective_p_value = util.compute_p_value(final_interval, etaTX, etaT_Sigma_eta)
    print(f"test-stat: {etaTX}, p-value:", selective_p_value)
    return selective_p_value


def oc_test(final_model, Xs, Xt, Sigma, K, device, _=None):
    # global final_model

    ns = Xs.shape[0]
    nt = Xt.shape[0]

    d = Xs.shape[1]
    n = ns + nt

    Xs_torch = torch.from_numpy(Xs).double().to(device)
    Xt_torch = torch.from_numpy(Xt).double().to(device)

    with torch.no_grad():
        xs_hat = final_model.extract_feature(Xs_torch).cpu().numpy()
        xt_hat = final_model.extract_feature(Xt_torch).cpu().numpy()
    
    Xs_vec = vec(Xs)
    Xt_vec = vec(Xt)
    X_vec = np.vstack((Xs_vec, Xt_vec))
    X_origin = np.vstack((Xs, Xt))
    X_transformed = np.vstack((xs_hat, xt_hat))

    initial_centroids_obs, labels_all_obs, members_all_obs = kmeans(X_transformed, K)
    try:
        a, b, eta_tmp, etaTX, etaT_Sigma_eta, c1, c2, c1_obs, c2_obs, sign = test_statistic(X_vec, Xt, ns, nt, d, K, Sigma, labels_all_obs).values()    
    except Exception as e:
        print("test statistic is none", e) 
        return None
    a_2d = a.reshape(n, d)
    b_2d = b.reshape(n, d)

    np_wdgrl = None# operations.convert_network_to_numpy(final_model.encoder)
    # final_model.encoder = final_model.encoder.to(device)



    final_interval = overconditioning(final_model, X_origin,ns, eta_tmp, a_2d, b_2d,np_wdgrl, K, initial_centroids_obs, labels_all_obs,z=etaTX, X_=X_transformed)
    # final_interval = [(-np.inf, np.inf)]
    selective_p_value = util.compute_p_value(final_interval, etaTX, etaT_Sigma_eta)
    print(f"test-stat: {etaTX}, p-value:", selective_p_value)
    return selective_p_value


def naive_test(Xs, Xt, Sigma, K, device, _=None):
    global final_model

    ns = Xs.shape[0]
    nt = Xt.shape[0]

    d = Xs.shape[1]
    n = ns + nt

    Xs_torch = torch.from_numpy(Xs).double().to(device)
    Xt_torch = torch.from_numpy(Xt).double().to(device)

    with torch.no_grad():
        xs_hat = final_model.extract_feature(Xs_torch).cpu().numpy()
        xt_hat = final_model.extract_feature(Xt_torch).cpu().numpy()
    
    Xs_vec = vec(Xs)
    Xt_vec = vec(Xt)
    X_vec = np.vstack((Xs_vec, Xt_vec))
    X_origin = np.vstack((Xs, Xt))
    X_transformed = np.vstack((xs_hat, xt_hat))

    initial_centroids_obs, labels_all_obs, members_all_obs = kmeans(X_transformed, K)
    try:
        a, b, eta_tmp, etaTX, etaT_Sigma_eta, c1, c2, c1_obs, c2_obs, sign = test_statistic(X_vec, Xt, ns, nt, d, K, Sigma, labels_all_obs).values()    
    except Exception as e:
        print("test statistic is none", e) 
        return None
    # a_2d = a.reshape(n, d)
    # b_2d = b.reshape(n, d)

    final_interval = [(-np.inf, np.inf)]
    selective_p_value = util.compute_p_value(final_interval, etaTX, etaT_Sigma_eta)
    print(f"test-stat: {etaTX}, p-value:", selective_p_value)
    return selective_p_value


def permu_test(final_model, Xs, Xt, Sigma, K, device, _=None):

    ns = Xs.shape[0]
    nt = Xt.shape[0]

    d = Xs.shape[1]
    n = ns + nt

    Xs_torch = torch.from_numpy(Xs).double().to(device)
    Xt_torch = torch.from_numpy(Xt).double().to(device)

    with torch.no_grad():
        xs_hat = final_model.extract_feature(Xs_torch).cpu().numpy()
        xt_hat = final_model.extract_feature(Xt_torch).cpu().numpy()
    
    Xs_vec = vec(Xs)
    Xt_vec = vec(Xt)
    X_vec = np.vstack((Xs_vec, Xt_vec))
    X_origin = np.vstack((Xs, Xt))
    X_transformed = np.vstack((xs_hat, xt_hat))

    initial_centroids_obs, labels_all_obs, members_all_obs = kmeans(X_transformed, K)
    try:
        a, b, eta_tmp, etaTX, etaT_Sigma_eta, c1, c2, c1_obs, c2_obs, sign = test_statistic(X_vec, Xt, ns, nt, d, K, Sigma, labels_all_obs).values()
        
    except Exception as e:
        print("test statistic is none", e) 
        return None

    permutation_test_pvalue = permutation_test(Xt, c1_obs, c2_obs, test_statistic_permutationtest, ns = ns, nt = nt, d = d)["p_value"]
    print("permutation test p-value:", permutation_test_pvalue)
    return permutation_test_pvalue

import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run iterations from start to end index")
    parser.add_argument("start", type=int, nargs="?", default=1, help="Start iteration index (default: 0)")
    parser.add_argument("end", type=int, nargs="?", default=1, help="End iteration index (inclusive, default: 1)")

    list_p_values = []
    # iteration = 24

    args = parser.parse_args()


    # for i in range(args.start, args.end + 1):
    #     print(f"\n--- Iteration {i}/{args.end} ---")
    #     p_value = run(mu_s, mu_t, K, device, i)
    #     if p_value is not None:
    #         list_p_values.append(p_value)

    # print("Running time:", time.time() - st, "(s)")
    # underalpha = sum(1 for p in list_p_values if p <= 0.05)
    # print('\nFalse positive rate:', underalpha/len(list_p_values), 'out of', len(list_p_values))

    # # Kiểm định thống kê
    # kstest = stats.kstest(list_p_values, 'uniform')
    # print(kstest)
    # Hiển thị histogram
    # plt.hist(list_p_values)
    # plt.show()
    # plt.savefig('logs/selective_inference_log/p_values_histogram.png')


    # with open('logs/selective_inference_log/p_values.txt', 'a') as f:

    #     f.write(f"\nFalse positive rate: {underalpha/len(list_p_values)} out of {len(list_p_values)}\n")
    #     f.write(f"\nKS test statistic: {kstest.statistic}, p-value: {kstest.pvalue}\n")
