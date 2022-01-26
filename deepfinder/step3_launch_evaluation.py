import argparse

import sys

import os
import utils.objl as ol

import configparser

# parse path to config
parser = argparse.ArgumentParser()
parser.add_argument("--path_config", type=str, help="path to the config.ini of the experiments")
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(path_config + 'config_loc_class.ini')

# load config information
models = config['trained_DF_weights']['models'].split()
num_epochs_list = config['trained_DF_weights']['num_epochs'].split()
test_tomos = config['trained_DF_weights']['test_tomos'].split()


for test_tomo in test_tomos:
    for model, num_epochs in zip(models, num_epochs_list):
        path = 'results/'
        
        objl_path = path + model + '/tomo9_'+test_tomo+'_2021_'+num_epochs+'epoch_bin1_objlist_thr.xml'

        objl = ol.read_xml(objl_path)
        # Then, we convert the predicted object list into a text file, as needed by the SHREC'21 evaluation script:
        
        class_name = {0: "0", 1: "4V94", 2: "4CR2", 3: "1QVR", 4: "1BXN", 5: "3CF3", 6: "1U6G",
                      7: "3D2F", 8: "2CG9", 9: "3H84", 10: "3GL1", 11: "3QM1", 
                      12: "1S3X", 13: "5MRC",  14: "vesicle", 15: "fiducial"}

        file = open(path + model+'/particle_locations_tomo9_'+test_tomo+'_2021.txt', 'w+')

        for p in range(0,len(objl)):
            x   = int( objl[p]['x'] )
            y   = int( objl[p]['y'] )
            z   = int( objl[p]['z'] )
            lbl = int( objl[p]['label'] )
            file.write(class_name[lbl]+' '+str(x)+' '+str(y)+' '+str(z)+'\n')
        file.close()

        
        os.system('python data/shrec2021_extended_dataset/misc/eval.py -s '+path+model+'/particle_locations_tomo9_'+test_tomo+'_2021.txt -t data/shrec2021_extended_dataset/model_9/faket/ -o '+path+model+'/report_'+model+'_'+test_tomo+'_bin1_2021.txt')
