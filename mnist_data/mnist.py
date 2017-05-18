# -*- coding:utf-8 -*-

import urllib.request
import os.path

import numpy as np
import gzip
from PIL import Image

url_base = "http://yann.lecun.com/exdb/mnist/"
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}
dataset_dir = os.path.dirname(os.path.abspath(__file__))

def _download(filename, verbose=False):
    file_path = dataset_dir + "/mnist/" + filename
    if os.path.exists(file_path):
        if verbose:
            return print("alredady exist")
        else: return
    print("Downloading " + filename + " ... ")
    urllib.request.urlretrieve(url_base + filename, file_path)
    print("Done")

def download_mnist():
    for v in key_file.values():
        for v in key_file.values():
            _download(v)

def load_mnist(filename, img_size=784):
    file_path = dataset_dir + "/mnist/" + filename
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, img_size)

def example():
    download_mnist()

    img = load_mnist(key_file["test_img"])
    label = load_mnist(key_file["test_label"], 1)
    for i in range(3):
        print(label[0+i+20])
        img1 = img[8+i+20].reshape(28, 28)
        pil_img = Image.fromarray(np.uint8(img1))
        pil_img.show()

if __name__ == "__main__":
    example()
