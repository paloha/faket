import os
import sys
import faket.deepfinder.utils.objl as ol


def launch_evaluation(test_tomogram, test_tomo_idx, num_epochs, label_map_path, out_path):
    
    part_file_name = f'{out_path}tomo{test_tomo_idx}_{test_tomogram}_2021_{num_epochs}epoch'
    objl = ol.read_xml(f'{part_file_name}_bin1_objlist_thr.xml')
    particle_list = f'{out_path}particle_locations_tomo_{test_tomogram}_.txt'
    
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
    eval_script = 'data/shrec2021_extended_dataset/misc/eval.py'
    args = [
        f'-s {particle_list}',
        f'-t data/shrec2021_extended_dataset/model_9/faket/',
        f'-o {part_file_name}_bin1_2021.txt'
    ]
    os.system(f'{interpreter} {eval_script} {" ".join(args)}')

