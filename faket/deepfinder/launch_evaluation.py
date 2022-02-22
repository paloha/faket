import os
import sys
import argparse
from pathlib import Path
import utils.objl as ol


#def launch_evaluation(test_tomogram, test_tomo_idx, num_epochs, label_map_path, out_path):

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_tomogram", type=str, 
                        help="tomogram to be segmented", default="baseline")
    parser.add_argument("--test_tomo_idx", type=str, 
                        help="folder index of test tomogram")
    parser.add_argument("--num_epochs", type=str, 
                        help="number of epochs deep finder was trained")
    parser.add_argument("--label_map_path", type=str, 
                        help="path to the folder of the label map that results from segmentation")
    parser.add_argument("--out_path", type=str, 
                        help="out path for the xml files resulting from clustering")
    args = parser.parse_args()

    part_file_name = f'{args.out_path}tomo{args.test_tomo_idx}_{args.test_tomogram}_2021_{args.num_epochs}epoch'
    objl = ol.read_xml(f'{part_file_name}_bin1_objlist_thr.xml')
    particle_list = f'{args.out_path}particle_locations_tomo_{args.test_tomogram}_.txt'
    
    # convert the predicted object list into a text file, as needed by the SHREC'21 evaluation script:
       
    class_name = {
        0: "0", 1: "4V94", 2: "4CR2", 3: "1QVR", 4: "1BXN", 5: "3CF3", 6: "1U6G",
        7: "3D2F", 8: "2CG9", 9: "3H84", 10: "3GL1", 11: "3QM1", 12: "1S3X", 
        13: "5MRC",  14: "vesicle", 15: "fiducial"
    }
    
    with open(particle_list, 'w+') as file:
        for p in range(0, len(objl)):
            x = int(objl[p]['x'])
            y = int(objl[p]['y'])
            z = int(objl[p]['z'])
            lbl = int(objl[p]['label'])
            file.write(f'{class_name[lbl]} {x} {y} {z}\n')
    
    interpreter = 'python'
    eval_script = Path('data/shrec2021_extended_dataset/misc/eval.py')
    test_tomogram_folder = Path('data/shrec2021_extended_dataset/model_9/faket/')
    args = [
        f'-s {particle_list}',
        f'-t {test_tomogram_folder}',
        f'-o {part_file_name}_bin1_2021.txt'
    ]
    os.system(f'{interpreter} {eval_script} {" ".join(args)}')

