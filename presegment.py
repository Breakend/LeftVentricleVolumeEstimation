import Image
import cv2
import math
import numpy as np
import yaml
import dicom
from data_utils import MRIDataIterator, convert_age, randword
from segment import calc_rois

def main():
    print("Creating data iterator...")
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    train_dir = cfg['dataset_paths']['train_data']
    train_labels = cfg['dataset_paths']['train_labels']

    mri_iter = MRIDataIterator(train_dir, train_labels)
    outputdir = './testpreproc/'
    for patient_index, patient_slices in mri_iter.frames.iteritems():
        slices_locations_to_names = {}
        i = 0
        for sax_set in patient_slices:
            slices_locations_to_names[int(dicom.read_file(sax_set[0]).SliceLocation)] = i
            i += 1
        median_array = slices_locations_to_names.keys()
        median_array.sort()
        values_closest_to_middle = []
        if len(median_array) > 1:
            middle_value = (median_array[-1] + median_array[0])/2
            for val in median_array:
                if math.sqrt((val - middle_value)**2) < 25:
                    values_closest_to_middle.append(val)
        else:
             middle_value = median_array[0]
             values_closest_to_middle.append(median_array[0])

        z = 0
        values = []
        for proposed_median_value in values_closest_to_middle:
            median_index = slices_locations_to_names[proposed_median_value]
            sax_set = patient_slices[median_index]
            time_series = []
            for path in sax_set:
                f = dicom.read_file(path)
                gender = f.PatientsSex
                age = convert_age(f.PatientsAge)
                img = mri_iter.preproc(f.pixel_array.astype(np.float32) / np.max(f.pixel_array), 64, f.PixelSpacing, True, False)
                time_series.append(img)
            values.append(time_series)
            z +=1
        data_array = np.array(values)
        rois,circles = calc_rois(data_array)
        i = 0
        import pdb; pdb.set_trace()
        new_set = []
        for sax_set in data_array:
            center_point, radius = circles[i]
            new_time_series = []
            for img in sax_set:
                # make it square
                crop_img = img[center_point[0]-40:center_point[0]+40, center_point[1]-60:center_point[1]+20]
                new_time_series.append(crop_img)
            new_set.append(new_time_series)
                
        new_data_array = np.array(new_set)
        im = Image.fromarray(new_data_array[0][0]).convert('RGB')
        im.save('examples/' + randword() +'.png')

main()
