import os
import sys
import argparse
import utils.objl as ol
from os.path import join as pj

if __name__ == '__main__':
    
    # Parse arguments
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
    
    identifier_fname = f'epoch{int(args.num_epochs):03d}_2021_model_{args.test_tomo_idx}_{args.test_tomogram}_bin2'
    
    # Path to the xml containing the clustering results
    xml_fname = f'{identifier_fname}_objlist_thr.xml'
    
    # Path to the txt file containing the predicted particle locations (for SHREC eval script)
    particle_list_fname = f'{identifier_fname}_particles.txt'
    
    # Desired path of the evaluation output txt file (evaluated using SHREC eval script)
    out_fname = f'{identifier_fname}_evaluation.txt'

    # Map for conversion of the predicted object list into a text file, as needed by the SHREC'21 evaluation script:
    class_name = {
        0: "0", 1: "4V94", 2: "4CR2", 3: "1QVR", 4: "1BXN", 5: "3CF3", 6: "1U6G",
        7: "3D2F", 8: "2CG9", 9: "3H84", 10: "3GL1", 11: "3QM1", 12: "1S3X",
        13: "5MRC",  14: "vesicle", 15: "fiducial"}
    
    # Build the particle list and output it as a txt file
    objl = ol.read_xml(pj(args.out_path, xml_fname))
    with open(pj(args.out_path, particle_list_fname), 'w+') as file:
        for p in range(0, len(objl)):
            x = int(objl[p]['x'])
            y = int(objl[p]['y'])
            z = int(objl[p]['z'])
            lbl = int(objl[p]['label'])
            file.write(f'{class_name[lbl]} {x} {y} {z}\n')
    
    # Run the original SHREC'21 evaluation script
    interpreter = 'python'
    eval_script = pj('data', 'shrec2021_extended_dataset', 'misc', 'eval.py')
    test_tomogram_folder = pj('data', 'shrec2021_extended_dataset', 'model_9', 'faket')
    args = [
        f'-s {pj(args.out_path, particle_list_fname)}',  # Predicted
        f'-t {test_tomogram_folder}',  #  Ground truth
        f'-o {pj(args.out_path, out_fname)}'  # Output
    ]
    os.system(f'{interpreter} {eval_script} {" ".join(args)}')
