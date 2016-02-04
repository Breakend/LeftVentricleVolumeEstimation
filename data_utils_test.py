import unittest
import dicom
import numpy as np

from matplotlib import pyplot as plt
from data_utils import MRIDataIterator
from random import randint

class TestDataUtilities(unittest.TestCase):
    def setUp(self):
        # TODO: this is super inefficient and dumb, rewrite so I don't read in entire space of all sax images
        self.mriIter = MRIDataIterator("/Users/Breakend/Documents/datasets/sciencebowl2015/train", "/Users/Breakend/Documents/datasets/sciencebowl2015/train.csv")

    def test_preprocessing(self):
        """ This is a visual test, sample a random image and preprocess it, view
            the result"""
        dicom_image_path = self.mriIter.frames[randint(1,599)][0][0]
        f = dicom.read_file(dicom_image_path)
        plt.figure(figsize=(10,3.6))
        plt.subplot(131)
        plt.imshow(f.pixel_array)
        img = self.mriIter.preproc(f.pixel_array.astype(np.float32) / np.max(f.pixel_array), 64, f.PixelSpacing)
        plt.subplot(132)
        plt.imshow(img)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01, left=0.05,
                    right=0.99)
        plt.show()


if __name__ == '__main__':
    unittest.main()
