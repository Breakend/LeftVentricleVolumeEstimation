"""Preprocessing script.

Note, much of this was taken from: https://raw.githubusercontent.com/dmlc/mxnet/master/example/kaggle-ndsb2/Preprocessing.py
"""
import os
import csv
import sys
import random
import scipy
import numpy as np
import dicom
from skimage import io, transform

# Extra, non-specific data utilities
def one_hot(size, value):
    return np.eye(size)[value]

class MRIDataIterator(object):
    """ Iterates over the fMRI scans and returns batches of test and validation
    data. Needed to load into memory one batch at a time."""

    def __init__(self, frame_root_path = None, label_path = None, percent_validation = .8):
        """Walk the directory and randomly split the data"""
        if frame_root_path:
            self.frames = self.get_frames(frame_root_path)
        if label_path:
            self.labels = self.get_label_map(label_path)
        self.current_iter_position = 0
        self.PATIENT_RANGE_INCLUSIVE = (1,500)
        self.last_training_index = int(percent_validation * self.PATIENT_RANGE_INCLUSIVE[1])
        self.histogram_bins = self.get_histogram_bins()

    def get_histogram_bins(self, attribute_name='SliceLocation'):
        dicom_images = []
        for key, value in self.frames.iteritems():
            for sax_frame in value:
                dicom_images.append(dicom.read_file(sax_frame[0]))
        numbers, bins = np.histogram([getattr(image, attribute_name) for image in dicom_images], 10)
        return bins

    def get_frames(self, root_path):
       """Get path to all the frame in view SAX and contain complete frames"""
       ret = {}
       for root, _, files in os.walk(root_path):
           if len(files) == 0 or not files[0].endswith(".dcm") or root.find("sax") == -1:
               continue
           prefix = files[0].rsplit('-', 1)[0]
           data_index = int(root.rsplit('/', 3)[1])
           fileset = set(files)
           expected = ["%s-%04d.dcm" % (prefix, i + 1) for i in range(30)]
           if all(x in fileset for x in expected):
               if data_index in ret:
                   ret[data_index].append([root + "/" + x for x in expected])
               else:
                   ret[data_index] = [[root + "/" + x for x in expected]]

       return ret

    def get_label_map(self, fname):
       labelmap = {}
       fi = open(fname)
       fi.readline()
       for line in fi:
           arr = line.split(',')
           labelmap[int(arr[0])] = [np.float32(x) for x in arr[1:]]
       return labelmap

    def preproc(self, img, size, pixel_spacing):
       """crop center and resize"""
        #    TODO: this is stupid, you could crop out the heart
        # But should test this
       if img.shape[0] < img.shape[1]:
           img = img.T
       # Standardize based on pixel spacing
       img = transform.resize(img, (int(img.shape[0]*(1.0/np.float32(pixel_spacing[0]))), int(img.shape[1]*(1.0/np.float32(pixel_spacing[1])))))
       # we crop image from center
       short_egde = min(img.shape[:2])
       yy = int((img.shape[0] - short_egde) / 2)
       xx = int((img.shape[1] - short_egde) / 2)
       crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
       # resize to 64, 64
       resized_img = transform.resize(crop_img, (size, size))
       resized_img *= 255.
       return resized_img.astype("float32")

    def has_more_training_data(self, index = None):
        if not index:
            index = self.current_iter_position
        return index <= self.last_training_index

    def has_more_validation_data(self, index):
        return index <= self.PATIENT_RANGE_INCLUSIVE[1]

    def get_median_bucket_data(self, index=None):
        """ Returns a batch in format (30, num_bins-1, 64, 64) which
            puts slices together, padding empty bucket layers with 0s"""

        # TODO: should incorporate slice thickness data in case a slice overlaps
        # several buckets?
        if not self.labels or not self.frames:
            raise ValueError("Frames or labels not set")
        if not index:
            index = self.current_iter_position
            self.current_iter_position += 1

        if self.PATIENT_RANGE_INCLUSIVE[0] > index > self.PATIENT_RANGE_INCLUSIVE[1]:
            raise ValueError("Index out of bounds for data.")

        #TODO: fix this
        patient_frames = self.frames[index]
        data_array = np.zeros((30, len(self.histogram_bins)-1, 64, 64))
        slices_locations_to_names = {}
        i = 0
        for sax_set in patient_frames:
            slices_locations_to_names[int(dicom.read_file(sax_set[0]).SliceLocation)] = i
            i += 1
        median_array = slices_locations_to_names.keys()
        median_array.sort()
        median_index = median_array[len(slices_locations_to_names.keys())/2]
        sax_set = patient_frames[median_index]
        for path in sax_set:
            f = dicom.read_file(path)
            img = self.preproc(f.pixel_array.astype(np.float32) / np.max(f.pixel_array), 64, f.PixelSpacing)
            bindex = np.clip(np.digitize([f.SliceLocation], self.histogram_bins, right=True)[0],0.,len(self.histogram_bins)-2)
            data_array[i][0][:][:] = np.array(img, dtype=np.float32)

        # data, systolic, diastolic
        # print("Systolic: {}".format()
        # print(np.array(np.float32(self.labels[index][0])))
        return data_array, np.array([np.float32(self.labels[index][0])]), np.array([np.float32(self.labels[index][1])]) #one_hot(600, self.labels[index][0]), one_hot(600, self.labels[index][0])#[ np.full(len(patient_frames), np.float32(x), dtype=np.float32) for x in self.labels[index]]


    def retrieve_data_batch_by_layer_buckets(self, index=None):
        """ Returns a batch in format (30, num_bins-1, 64, 64) which
            puts slices together, padding empty bucket layers with 0s"""

        # TODO: should incorporate slice thickness data in case a slice overlaps
        # several buckets?
        if not self.labels or not self.frames:
            raise ValueError("Frames or labels not set")
        if not index:
            index = self.current_iter_position
            self.current_iter_position += 1

        if self.PATIENT_RANGE_INCLUSIVE[0] > index > self.PATIENT_RANGE_INCLUSIVE[1]:
            raise ValueError("Index out of bounds for data.")

        #TODO: fix this
        patient_frames = self.frames[index]
        data_array = np.zeros((30, len(self.histogram_bins)-1, 64, 64))
        for sax_set in patient_frames:
            data = []
            i = 0
            for path in sax_set:
                f = dicom.read_file(path)
                img = self.preproc(f.pixel_array.astype(np.float32) / np.max(f.pixel_array), 64, f.PixelSpacing)
                bindex = np.clip(np.digitize([f.SliceLocation], self.histogram_bins, right=True)[0],0.,len(self.histogram_bins)-2)
                data_array[i][bindex][:][:] = np.array(img, dtype=np.float32)
                i += 1

        # data, systolic, diastolic
        # print("Systolic: {}".format()
        # print(np.array(np.float32(self.labels[index][0])))
        return data_array, np.array([np.float32(self.labels[index][0])]), np.array([np.float32(self.labels[index][1])]) #one_hot(600, self.labels[index][0]), one_hot(600, self.labels[index][0])#[ np.full(len(patient_frames), np.float32(x), dtype=np.float32) for x in self.labels[index]]

    def retrieve_data_batch_with_time_as_channel(self, index = None):
        """ Minibatched data retrieval of fMRI images, returns a numpy array
        of (num_sax_images x 30 x 64 x 64) and the equivalent label (arr, label)
        loaded into memory with the 30 being the channel related to fMRI slices
        and num_sax_images being the sequence images of the heart cycle
        to find systole and diastole"""
        if not self.labels or not self.frames:
            raise ValueError("Frames or labels not set")
        if not index:
            index = self.current_iter_position
            self.current_iter_position += 1

        if self.PATIENT_RANGE_INCLUSIVE[0] > index > self.PATIENT_RANGE_INCLUSIVE[1]:
            raise ValueError("Index out of bounds for data.")

        patient_frames = self.frames[index]
        data_array = np.zeros((len(patient_frames), 30, 64, 64))
        sax_index = 0
        for sax_set in patient_frames:
            data = []
            for path in sax_set:
                f = dicom.read_file(path)
                img = self.preproc(f.pixel_array.astype(np.float32) / np.max(f.pixel_array), 64.)
                data.append(img)
            data = np.array(data, dtype=np.float32)
            # data = data.reshape(data.size)
            data_array[sax_index][:][:][:] = data
            sax_index += 1

        return data_array, [ np.full(len(patient_frames), np.float32(x), dtype=np.float32) for x in self.labels[index]]

    # TODO: modify this for writing the validation labels
    def write_label_csv(self, fname, frames, label_map):
       fo = open(fname, "w")
       for lst in frames:
           index = int(lst[0].split("/")[3])
           if label_map != None:
               fo.write(label_map[index])
           else:
               fo.write("%d,0,0\n" % index)
       fo.close()
