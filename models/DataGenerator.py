"""
Definition of DataGenerator class which should encompass both the pix2pix
examples and CBRAIN.
"""
from .utils import *
import threading
from glob import glob


# To make generators thread safe for multithreading
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return ThreadsafeIter(f(*a, **kw))
    return g


class ThreadsafeIter(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    https://github.com/fchollet/keras/issues/1638
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):   # Py3
        with self.lock:
            return next(self.it)


def generator(a_imgs, b_imgs, bs, n_batches, shuffle=False):
    """First basic generator to work with facades dataset.
    """
    while True:
        for i in range(n_batches):
            x = a_imgs[i*bs:(i+1)*bs]
            y = b_imgs[i*bs:(i+1)*bs]
            yield x, y


class DataGenerator(object):
    def __init__(self, data_dir, img_size=256, bs=4):
        self.files = sorted(glob(data_dir + '*'))
        self.a_imgs, self.b_imgs = load_all_imgs(self.files, img_size)
        self.bs = bs
        self.n_samples = self.a_imgs.shape[0]
        self.n_batches = int(np.floor(self.n_samples / bs))
        self.gen = generator(self.a_imgs, self.b_imgs, bs, self.n_batches)
