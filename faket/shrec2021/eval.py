# This script originally comes from SHREC2021 challenge `misc/eval.py`.
# We (FakET) copied it and made only necessary changes such that we can 
# export the confusion matrices and stats as machine readable files. 
# Which was not possible without a code change. 

# We also fixed some bugs: 
# - bug with argparser bool flags
# - bug with handling exclusion of particles
# All our chagnes are commented with "# FakET".

# ---------------------------------------------------------------------
# Dependencies can be installed with the following command:
# pip install scikit-plot seaborn scikit-image scipy pycm numpy mrcfile
# Fill out argument defaults or pass the arguments with CLI parameters

import mrcfile as mrc
import numpy as np
import warnings
from pathlib import Path
from pycm import ConfusionMatrix
from pycm.pycm_output import table_print, stat_print
from pycm.pycm_param import SUMMARY_CLASS, SUMMARY_OVERALL
from scipy.spatial import distance
from skimage.morphology import dilation
import argparse

# To avoid mrcfile warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import scikitplot as skplt

# FakET change
import os
import json
from pycm.pycm_output import csv_matrix_print
# --------------

if __name__ == '__main__':

    # Script parameters
    parser = argparse.ArgumentParser(description='SHREC 2021 Cryo-ET evaluation script')

    parser.add_argument('-s', '--submission',   type=Path, default='',
                        help='Path (1) to a directory with submissions or (2) to a specific submission txt file')
    parser.add_argument('-t', '--tomo',         type=Path, default='../model_9',
                        help='Path to the folder with test tomogram ground truth')
    parser.add_argument('-o', '--output',       type=Path, default='results.txt',
                        help='Path to output file')
    parser.add_argument('--confusion', action='store_true',  # FakET change (if not provided, means False)
                        help='Whether to plot individual confusion matrices or not')
    parser.add_argument('--skip_4v94', action='store_true',  # FakET change (if not provided, means False)
                        help='Whether to skip 4V94 evaluation or not. True in SHREC Cryo-ET 2021 results.')
    parser.add_argument('--skip_vesicles', action='store_true',  # FakET change (if not provided, means False)
                        help='Whether to skip vesicles or not. True in SHREC Cryo-ET 2021 results.')
    args = parser.parse_args()

    # Prepare list of submissions to check, only if passed argument is a directory
    if args.submission.is_dir():
        submissions = list(args.submission.glob('**/*.txt'))
        submissions.sort()
        submission_names = [f'{s.parent.stem}_{s.name}' for s in submissions]
    elif args.submission.suffix == '.txt':
        submissions = [args.submission]
    else:
        raise ValueError(f'The passed submission argument should be either a .txt file or a directory with txt files.')

    # Conversion dicts
    classes = ['background', '4V94', '4CR2', '1QVR', '1BXN', '3CF3', '1U6G', '3D2F', 
               '2CG9', '3H84', '3GL1', '3QM1', '1S3X', '5MRC', 'vesicle', 'fiducial']
    num2pdb = {k: v for k, v in enumerate(classes)}
    pdb2num = {v: k for k, v in num2pdb.items()}

    # FakET bugfix (introducing variable excluded)
    excluded = []
    if args.skip_4v94:
        excluded.append('4V94')
        del num2pdb[pdb2num['4V94']]
        del pdb2num['4V94']
    if args.skip_vesicles:
        excluded.append('vesicle')
        del num2pdb[pdb2num['vesicle']]
        del pdb2num['vesicle']

    # Loading all ground truth particles
    gt_particles = [('0', 0, 0, 0)]  # start with a "background" particle
    with open(args.tomo / 'particle_locations.txt', 'rU') as f:
        for line in f:
            pdb_id, x, y, z, *_ = line.rstrip('\n').split()
            gt_particles.append((pdb_id, int(x), int(y), int(z)))

    n_gt_particles = len(gt_particles) - 1
    if args.skip_4v94:
        n_4v94_particle = len([p for p in gt_particles if p[0] == '4V94'])
        n_gt_particles -= n_4v94_particle
    
    # FakET - I am not sure why SHREC does not subtract also n_vesicles_particles
    # from n_gt_particles if args.skip_vesicles, maybe they do not count it as a particle??
    # Shall we add the following code then??
    # if args.skip_vesicles:
    #     n_vesicles = len([p for p in gt_particles if p[0] == 'vesicle'])
    #     n_gt_particles -= n_vesicles

    # load class mask and occupancy mask
    with mrc.open(args.tomo / 'occupancy_mask.mrc', permissive=True) as f:
        occupancy_mask = dilation(f.data)

    # # Debugging visualization
    # import napari
    # with napari.gui_qt():
    #     v = napari.Viewer()
    #     v.add_image(dilated)

    # FakET change - Overwrite previous evaluation.txt
    eval_fpath = os.path.join(args.output, 'evaluation.txt')
    if os.path.exists(eval_fpath):
        os.remove(eval_fpath)
    
    # Evaluate each submission
    for submission_i, submission in enumerate(submissions):
        gt_particles2 = gt_particles.copy()
        n_gt_particles2 = n_gt_particles
        classes2 = classes

        report = f'Processing submission #{submission_i}: {submission}\n'
        print(report)
        # FakET changes
        # with open(args.output, mode='a+') as f:
        #     f.write(report)
        with open(eval_fpath, mode='a+') as f:
                  f.write(report)

        # Load predicted particles and their classes
        predicted_particles = []
        with open(submission, 'rU') as f:
            for line in f:
                pdb, x, y, z, *_ = line.rstrip('\n').split()
                # if pdb != 'fiducial':  # BUG IN ORIGINAL EVAL, FIX BY FakET
                if pdb not in ['fiducial', 'vesicle']:  # background should be already excluded at clustering step
                    pdb = pdb.upper()
                predicted_particles.append((pdb, int(round(float(x))), int(round(float(y))), int(round(float(z)))))
        n_predicted_particles = len(predicted_particles)

        # Init of some vars for statistics
        n_clipped_predicted_particles = 0  # number of particles that were predicted to be outside of tomogram
        found_particles = [[] for _ in range(len(gt_particles2))]  # reported classes and distances for each GT particle

        # Go through each predicted particle
        for p_i, (p_pdb, *coordinates) in enumerate(predicted_particles):

            # Clamp coordinates to avoid out of bounds
            p_x, p_y, p_z = np.clip(coordinates, (0, 0, 0), (511, 511, 511))

            # Were coordinates out of bounds?
            if [p_x, p_y, p_z] != coordinates:
                n_clipped_predicted_particles += 1

            # Find ground truth particle at the predicted location
            p_gt_id = int(occupancy_mask[p_z, p_y, p_x])
            p_gt_pdb, p_gt_x, p_gt_y, p_gt_z = gt_particles2[p_gt_id]

            # Compute distance from predicted center to real center
            p_distance = np.abs(distance.euclidean((p_x, p_y, p_z), (p_gt_x, p_gt_y, p_gt_z)))

            # Register found particle, a class it is predicted to be and distance from predicted center to real center
            found_particles[p_gt_id].append((p_pdb, p_distance))

        # FakET change (use excluded variable) ------------------------------
        # if particle is in excluded, e.g. 4V94, remove all of them from both GT and predicted
        if excluded:
            for i in range(len(gt_particles2) - 1, -1, -1):
                if gt_particles2[i][0] in excluded:
                    del gt_particles2[i]
                    del found_particles[i]

        # original code from SHREC
        # if args.skip_4v94:
        #     for i in range(len(gt_particles2) - 1, 0, -1):
        #         if gt_particles2[i][0] == '4V94':
        #             del gt_particles2[i]
        #             del found_particles[i]

        # if args.skip_vesicles:
        #     for i in range(len(gt_particles2) - 1, -1, -1):
        #         if gt_particles2[i][0] == 'vesicle':
        #             del gt_particles2[i]
        #             del found_particles[i]
        # -----------------------------------------------------------------------

        # Compute localization statistics
        n_prediction_missed = len(found_particles[0])
        n_prediction_hit = sum([len(p) for p in found_particles[1:]])
        n_unique_particles_found = sum([int(p >= 1) for p in [len(p) for p in found_particles[1:]]])
        n_unique_particles_not_found = sum([int(p == 0) for p in [len(p) for p in found_particles[1:]]])
        n_unique_particle_with_multiple_hits = sum([int(p > 1) for p in [len(p) for p in found_particles[1:]]])

        localization_recall = n_unique_particles_found / n_gt_particles2
        localization_precision = n_unique_particles_found / n_predicted_particles
        localization_f1 = 1 / ((1/localization_recall + 1/localization_precision) / 2)
        localization_miss_rate = 1 - localization_recall
        localization_avg_distance = sum([p[0][1] for p in found_particles[1:] if len(p) > 0]) / n_unique_particles_found

        # Compute classification statistics and confusion matrix
        gt_particle_classes = np.asarray([pdb2num[p[0]] for p in gt_particles2[1:]], dtype=int)
        
        # BUG IN ORIGINAL EVAL SCRIPT, FIX BY FakET
        # predicted_particle_classes = np.asarray(
        #     [pdb2num[p[0][0]] if (p and p[0][0] != '4V94') else 0 
        #          for p in found_particles[1:]], 
        #     dtype=int)  # taking the first occurrence only
        
        predicted_particle_classes = np.asarray(  # If in excluded, prediction is set to 0 (background)
            [pdb2num[p[0][0]] if (p and p[0][0] not in excluded) else 0 
                 for p in found_particles[1:]], 
            dtype=int)  # taking the first occurrence only (FakET comment: WHY the first?)
        
        confusion_matrix = ConfusionMatrix(actual_vector=gt_particle_classes, predict_vector=predicted_particle_classes)
        confusion_matrix.relabel(num2pdb)

        # FakET changes (to make confusion matrix file names alike output file name)
        labels = [  # Sorted according to size
            'background', 
            '4V94', # Most probably excluded (depends on args)
            '1S3X', '3QM1', '3GL1', '3H84', '2CG9',  # small
            '3D2F', '1U6G', '3CF3', '1BXN', '1QVR',  # medium
            '4CR2', '5MRC', # large
            'vesicle', # Most probably excluded (depends on args)
            'fiducial']  
        labels = [p for p in labels if p not in excluded]
        # --------------
        
        if args.confusion:
            lut_classes = np.asarray(classes2)
            skplt.metrics.plot_confusion_matrix(lut_classes[gt_particle_classes],
                                                lut_classes[predicted_particle_classes],
                                                labels=labels, figsize=(20, 20), text_fontsize=18, 
                                                hide_zeros=True, title=submission.stem, hide_counts=True)
            # FakET changes (to make confusion matrix file names alike output file name)
            plt.savefig(os.path.join(args.output, 'confusion_matrix_plain.png'))
            # plt.savefig(str(args.output.parent / f'{submission.parent.name}_{submission.name}_plain_cm.png'))
            # --------------
            

            skplt.metrics.plot_confusion_matrix(lut_classes[gt_particle_classes],
                                                lut_classes[predicted_particle_classes],
                                                labels=labels, figsize=(20, 20), text_fontsize=18, 
                                                hide_zeros=True, title=submission.stem, hide_counts=False)
            # FakET changes (to make confusion matrix file names alike output file name)
            plt.savefig(os.path.join(args.output, 'confusion_matrix_numbers.png'))
            # plt.savefig(str(args.output.parent / f'{submission.parent.name}_{submission.name}_numbers_cm.png'))
            # --------------

        # Prepare confusion matrix prints
        confusion_matrix_table = table_print(confusion_matrix.classes, confusion_matrix.table)
        confusion_matrix_stats = stat_print(confusion_matrix.classes, confusion_matrix.class_stat,
                                            confusion_matrix.overall_stat, confusion_matrix.digit,
                                            SUMMARY_OVERALL, SUMMARY_CLASS)

        # Format confusion matrix and stats
        confusion_matrix_table = '\t'.join(confusion_matrix_table.splitlines(True))
        confusion_matrix_stats = '\t'.join(confusion_matrix_stats.splitlines(True))

        # Construct a report and write it
        report = f'\n\t### Localization\n' \
                 f'\tSubmission has {n_predicted_particles} predicted particles\n' \
                 f'\tTomogram has {n_gt_particles2} particles\n' \
                 f'\tTP: {n_unique_particles_found} unique particles found\n' \
                 f'\tFP: {n_prediction_missed} predicted particles are false positive\n' \
                 f'\tFN: {n_unique_particles_not_found} unique particles not found\n' \
                 f'\tThere was {n_unique_particle_with_multiple_hits} particles that had more than one prediction\n' \
                 f'\tThere was {n_clipped_predicted_particles} predicted particles that were outside of tomo bounds\n' \
                 f'\tAverage euclidean distance from predicted center to ground truth center: {localization_avg_distance}\n' \
                 f'\tRecall: {localization_recall:.5f}\n' \
                 f'\tPrecision: {localization_precision:.5f}\n' \
                 f'\tMiss rate: {localization_miss_rate:.5f}\n' \
                 f'\tF1 score: {localization_f1:.5f}\n' \
                 f'\n\t### Classification\n' \
                 f'\t{confusion_matrix_table}\n' \
                 f'\t{confusion_matrix_stats}\n\n\n'
        
        # FakET changes (to unify the file names and export meaningful files):
        
        # print(report)
        # with open(args.output, mode='a+') as f:
        #    f.write(report)
        # confusion_matrix.save_html(str(args.output.parent / f'{submission.parent.name}_{submission.name}_confusion_matrix'))
        with open(os.path.join(args.output, 'evaluation.txt'), mode='a+') as f:
            f.write(report)
       
        # Saving human-readable confusion matrix and stats
        confusion_matrix.save_html(os.path.join(args.output, 'evaluation'))
        
        # Saving computer readable stats for localization and classification
        with open(os.path.join(args.output, 'evaluation.json'), 'w') as stat:
            json.dump({
                'localization': {
                    'predicted_particles': n_predicted_particles,
                    'expected_particles': n_gt_particles2,
                    'TP': n_unique_particles_found,
                    'FP': n_prediction_missed,
                    'FN': n_unique_particles_not_found,
                    'recall': localization_recall,
                    'Precision': localization_precision,
                    'Miss rate': localization_miss_rate,
                    'F1 score': localization_f1,
                    'avg_euclid_dist_from_real_center': localization_avg_distance,
                    'n_particles_with_multiple_preds': n_unique_particle_with_multiple_hits,
                    'n_particles_outside_bounds': n_clipped_predicted_particles},
                'classification': {
                    'overall': confusion_matrix.overall_stat,
                    'per_class': confusion_matrix.class_stat}},
                stat, indent=4)
            
        # Saving confusion matrix in tabular form
        with open(os.path.join(args.output, 'confusion_matrix.csv'), 'w') as csv:
            csv.write(csv_matrix_print(labels, confusion_matrix.table, header=True))
        
        # Saving normalized confusion matrix in tabular form
        with open(os.path.join(args.output, 'confusion_matrix_normalized.csv'), 'w') as csv:
            csv.write(csv_matrix_print(labels, confusion_matrix.normalized_table, header=True))
        # --------------
        