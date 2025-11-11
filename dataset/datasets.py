import h5py
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath


    def read_images_labels(self,images_filepath, labels_filepath):        
        # --- Read labels ---
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            labels = np.frombuffer(file.read(), dtype=np.uint8)

        # --- Read images ---
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            image_data = np.frombuffer(file.read(), dtype=np.uint8)

        # Reshape into (n_samples, rows, cols)
        images = image_data.reshape(size, rows, cols)

        # Resize each image to 16x16 and flatten
        resized_images = []
        for img in images:
            pil_img = Image.fromarray(img)  # 28x28 grayscale
            pil_resized = pil_img.resize((16, 16), Image.Resampling.LANCZOS)  # updated constant
            resized_images.append(np.array(pil_resized).flatten())  # shape (256,)

        resized_images = np.array(resized_images, dtype=np.uint8)  # shape (n_samples, 256)
        
        return resized_images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)   

def getUSPS():
    path = "dataset\\USPS\\usps.h5"

    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        Xtrain = train.get('data')[:]
        ytrain = train.get('target')[:]
        test = hf.get('test')
        Xtest = test.get('data')[:]
        ytest = test.get('target')[:]
    return {
        "Xtrain": Xtrain,
        "ytrain": ytrain,
        "Xtest": Xtest,
        "ytest": ytest,
    }

def getMNIST():
    training_images_filepath = 'dataset/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte'
    training_labels_filepath = 'dataset/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    test_images_filepath = 'dataset/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    test_labels_filepath = 'dataset/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (X_train, y_train), (X_test, y_test) = mnist_dataloader.load_data()

    return {
        "Xtrain": X_train,
        "ytrain": y_train,
        "Xtest": X_test,
        "ytest": y_test,
    }
if __name__ == "__main__":
    USPS = getUSPS()
    MNIST = getMNIST()

    X_source_train = MNIST["Xtrain"]
    y_source_train = MNIST["ytrain"]

    X_target_train = USPS["Xtrain"]
    y_target_train = USPS["ytrain"]

    selected_labels = [0, 1, 3, 8]

    target_mask = np.isin(y_target_train, selected_labels)
    source_mask = np.isin(y_source_train, selected_labels)

    X_target_train = X_target_train[target_mask]
    y_target_train = y_target_train[target_mask]

    X_source_train = X_source_train[source_mask]
    y_source_train = y_source_train[source_mask]

    scaler = MinMaxScaler()
    X_target_train_norm = scaler.fit_transform(X_target_train)
    X_source_train_norm = scaler.fit_transform(X_source_train)

    print("Source label counts:", dict(Counter(y_source_train)))
    print("Target label counts:", dict(Counter(y_target_train)))
