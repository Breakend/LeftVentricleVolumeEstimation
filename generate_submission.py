#!/usr/bin/python

import yaml
import numpy as np
import csv
import lasagne
import theano.tensor as T
import theano
import os

from data_utils import MRIDataIterator
from convnets import build_cnn

def compose_prediction_functions(scope):
    input_var = T.tensor4(scope + 'inputs')
    network = build_cnn(input_var, 20)

    prediction = lasagne.layers.get_output(network)
    prediction_fn = theano.function([input_var], prediction)
    return network, prediction_fn

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
    validation_dir = cfg['dataset_paths']['validation_data']
    sample_submission_path = cfg['dataset_paths']['sample_submission']

mriIter = MRIDataIterator(validation_dir)
systolic_network, systolic_prediction_fn = compose_prediction_functions('sys')
diastolic_network, diastolic_prediction_fn = compose_prediction_functions('dia')

if os.path.exists('model-sys.npz'):
    with np.load('model-sys.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(systolic_network, param_values)

if os.path.exists('model-dia.npz'):
    with np.load('model-dia.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(diastolic_network, param_values)

#TODO: Abstract data retrieval so it applies to validation (i.e. get bounds from number of folders in dataset path)
index = 501
sub_systole = {}
sub_diastole = {}
while mriIter.has_more_data(index):
    print("Index %s" % index)
    inputs = mriIter.get_median_bucket_data(index, 20, return_labels=False)
    i = 0
    while i < 20:
        sub_systole[index+i] = np.cumsum(systolic_prediction_fn(inputs)[i])
        sub_diastole[index+i] = np.cumsum(diastolic_prediction_fn(inputs)[i])
    index += 20

# write to submission file
print('Writing submission to file...')
fi = csv.reader(open(sample_submission_path))
f = open('submission.csv', 'w')
fo = csv.writer(f, lineterminator='\n')
fo.writerow(fi.next())
for line in fi:
    idx = line[0]
    key, target = idx.split('_')
    key = int(key)
    out = [idx]
    # want it to throw an error if the key doesn't exist
    if target == 'Diastole':
        out.extend(list(sub_diastole[key]))
    else:
        out.extend(list(sub_systole[key]))
    fo.writerow(out)

f.close()
