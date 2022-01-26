# This script computes recall, precision and f1-score for each object class, and prints out the result in log files.
# The evaluation is based on a script used for the challenge "SHREC 2019: Classification in cryo-electron tomograms"

# This script needs python3 and additional packages (see evaluate.py), as it was coded by SHREC'19 organizers. The 
# scores have been published in Gubins & al., "SHREC'19 track: classification in cryo-electron tomograms", 2019

import sys

import os
import utils.objl as ol

import configparser

config = configparser.ConfigParser()
config.read('deepfinder/config_loc_class.ini')

# load config information
models = config['trained_DF_weights']['models'].split()
num_epochs_list = config['trained_DF_weights']['num_epochs'].split()
test_tomos = config['trained_DF_weights']['test_tomos'].split()


for test_tomo in test_tomos:
    for model, num_epochs in zip(models, num_epochs_list):
        path = 'data/deepfinder/localization_classification/'
        
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

        os.system('python deepfinder/eval.py -s '+path+model+'/particle_locations_tomo9_'+test_tomo+'_2021.txt -t data/shrec2021_extended_dataset/model_9/faket/ -o '+path+model+'/report_'+model+'_'+test_tomo+'_bin1_2021.txt')
