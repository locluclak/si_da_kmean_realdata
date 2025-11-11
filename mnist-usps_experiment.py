import numpy as np
from models.wdgrl import WDGRL
import pandas as pd
import statisticaltest_cluster_da as cda
from scipy.linalg import block_diag
import cv2

from torchvision import datasets, transforms
from dataset.datasets import getUSPS


def resize_to_16x16(X):
    X_resized = np.zeros((X.shape[0], 16, 16), dtype=np.float32)
    for i in range(X.shape[0]):
        X_resized[i] = cv2.resize(X[i], (16, 16), interpolation=cv2.INTER_AREA)
    return X_resized



transform = transforms.ToTensor()
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Convert to NumPy
X_test  = test_dataset.data.numpy().astype("float64")
y_mnist_test  = test_dataset.targets.numpy()
X_test  /= 255.0

noise_test  = np.random.normal(0, 0.01, X_test.shape ).astype("float64")
X_test_noisy  = X_test  + noise_test

X_test_noisy  = np.clip(X_test_noisy, 0.0, 1.0)
X_mnist_test_16  = resize_to_16x16(X_test_noisy)


USPS = getUSPS()
X_usps_test = USPS["Xtest"]
y_usps_test = USPS["ytest"]

X_source = X_mnist_test_16.reshape(-1, 256)
X_target = X_usps_test.reshape(-1, 256)

selected_label = [0, 1, 3, 8]

X_source = X_mnist_test_16.reshape(-1, 256)
y_source = y_mnist_test

X_target = X_usps_test.reshape(-1, 256)
y_target = y_usps_test

# Now compute masks and filter (lengths match)
source_indices = np.isin(y_source, selected_label)
target_indices = np.isin(y_target, selected_label)

X_source = X_source[source_indices]
y_source = y_source[source_indices]
X_target = X_target[target_indices]
y_target = y_target[target_indices]


d = X_source.shape[1]
K = len(selected_label)



device = "cpu"

# ==== WDGRL model ====
final_model = WDGRL(
    input_dim=d,
    encoder_hidden_dims=[300,100],
    critic_hidden_dims=[100],
    alpha1=0.0005,
    alpha2=0.000005,
    seed=42,
)


final_model.load_model("logs\\20251111-220822-mnist-usps")

# X_source = pd.read_csv("obesitylevel/gender0_test.csv")
# X_target = pd.read_csv("obesitylevel/gender1_test.csv")
# to numpy


# X_source = X_source.to_numpy(dtype=np.float64)
# X_target = X_target.to_numpy(dtype=np.float64)

iterations = 30
log_dir = "logs/selective_inference_log/log_realdata/mnist_usps"

for i in range(iterations):
    X_source = X_source[np.random.choice(X_source.shape[0], 150, replace=False)]
    X_target = X_target[np.random.choice(X_target.shape[0], 100, replace=False)]

    ns = X_source.shape[0]
    nt = X_target.shape[0]

    cov_source = np.cov(X_source, rowvar=False, bias=False)
    cov_target = np.cov(X_target, rowvar=False, bias=False)

    cov_block = block_diag(cov_source, cov_target)
    Ins = np.eye(ns)
    Int = np.eye(nt)
    cov_vecXs = np.kron(cov_source, Ins)
    cov_vecXt = np.kron(cov_target, Int)
    Sigma = block_diag(cov_vecXs, cov_vecXt)

    oc_pvalue = cda.oc_test(final_model=final_model, Xs=X_source, Xt=X_target, Sigma=Sigma, K=K, device=device)
    with open(f"{log_dir}/obesity_pvalue_oc.txt", "a") as f:
        f.write(f"{oc_pvalue}\n")
    
    permutation_pvalue = cda.permu_test(final_model=final_model, Xs=X_source, Xt=X_target, Sigma=Sigma, K=K, device=device)
    with open(f"{log_dir}/obesity_pvalue_permutation.txt", "a") as f:
        f.write(f"{permutation_pvalue}\n")

    para_pvalue = cda.parametric_test(final_model=final_model, Xs=X_source, Xt=X_target, Sigma=Sigma, K=K, device=device)
    with open(f"{log_dir}/obesity_pvalue_para.txt", "a") as f:
        f.write(f"{para_pvalue}\n")

    print("Selective p-value after DA and clustering:", oc_pvalue)
    print("Parametric p-value after DA and clustering:", para_pvalue)
    print("Permutation p-value after DA and clustering:", permutation_pvalue)
