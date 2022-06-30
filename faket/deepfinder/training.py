# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

import gc
import os
import sys
import h5py
import time
import json
import models
import losses
import datetime
import numpy as np
from utils import core
import tensorflow as tf
from utils import common as cm
from os.path import join as pj
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support


class TargetBuilder(core.DeepFinder):
    def __init__(self):
        core.DeepFinder.__init__(self)

        self.remove_flag = False # if true, places '0' at object voxels, instead of 'lbl'.
                                 # Usefull in annotation tool, for removing objects from target

    # Generates segmentation targets from object list. Here macromolecules are annotated with their shape.
    # INPUTS
    #   objl: list of dictionaries. Needs to contain [phi,psi,the] Euler angles for orienting the shapes.
    #   target_array: 3D numpy array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
    #                 index order of array should be [z,y,x]
    #   ref_list: list of binary 3D arrays (expected to be cubic). These reference arrays contain the shape of macromolecules ('1' for 'is object' and '0' for 'is not object')
    #             The references order in list should correspond to the class label
    #             For ex: 1st element of list -> reference of class 1
    #                     2nd element of list -> reference of class 2 etc.
    # OUTPUT
    #   target_array: 3D numpy array. '0' for background class, {'1','2',...} for object classes.
    def generate_with_shapes(self, objl, target_array, ref_list):
        """Generates segmentation targets from object list. Here macromolecules are annotated with their shape.

        Args:
            objl (list of dictionaries): Needs to contain [phi,psi,the] Euler angles for orienting the shapes.
            target_array (3D numpy array): array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
                index order of array should be [z,y,x]
            ref_list (list of 3D numpy arrays): These reference arrays are expected to be cubic and to contain the shape of macromolecules ('1' for 'is object' and '0' for 'is not object')
                The references order in list should correspond to the class label.
                For ex: 1st element of list -> reference of class 1; 2nd element of list -> reference of class 2 etc.

        Returns:
            3D numpy array: Target array, where '0' for background class, {'1','2',...} for object classes.
        """
        self.check_arguments(objl, target_array, ref_list)

        N = len(objl)
        dim = target_array.shape
        for p in range(len(objl)):
            self.display('Annotating object ' + str(p + 1) + ' / ' + str(N) + ' ...')
            lbl = int(objl[p]['label'])
            x = int(objl[p]['x'])
            y = int(objl[p]['y'])
            z = int(objl[p]['z'])
            phi = objl[p]['phi']
            psi = objl[p]['psi']
            the = objl[p]['the']

            ref = ref_list[lbl - 1]
            centeroffset = np.int(np.floor(ref.shape[0] / 2)) # here we expect ref to be cubic

            # Rotate ref:
            if phi!=None and psi!=None and the!=None:
                ref = cm.rotate_array(ref, (phi, psi, the))
                ref = np.int8(np.round(ref))

            # Get the coordinates of object voxels in target_array
            obj_voxels = np.nonzero(ref == 1)
            x_vox = obj_voxels[2] + x - centeroffset #+1
            y_vox = obj_voxels[1] + y - centeroffset #+1
            z_vox = obj_voxels[0] + z - centeroffset #+1

            for idx in range(x_vox.size):
                xx = x_vox[idx]
                yy = y_vox[idx]
                zz = z_vox[idx]
                if xx >= 0 and xx < dim[2] and yy >= 0 and yy < dim[1] and zz >= 0 and zz < dim[0]:  # if in tomo bounds
                    if self.remove_flag:
                        target_array[zz, yy, xx] = 0
                    else:
                        target_array[zz, yy, xx] = lbl

        return np.int8(target_array)

    def check_arguments(self, objl, target_array, ref_list):
        self.is_list(objl, 'objl')
        self.is_3D_nparray(target_array, 'target_array')
        self.is_list(ref_list, 'ref_list')

    # Generates segmentation targets from object list. Here macromolecules are annotated with spheres.
    # This method does not require knowledge of the macromolecule shape nor Euler angles in the objl.
    # On the other hand, it can be that a network trained with 'sphere targets' is less accurate than with 'shape targets'
    # INPUTS
    #   objl: list of dictionaries.
    #   target_array: 3D numpy array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
    #                 index order of array should be [z,y,x]
    #   radius_list: list of sphere radii (in voxels).
    #             The radii order in list should correspond to the class label
    #             For ex: 1st element of list -> sphere radius for class 1
    #                     2nd element of list -> sphere radius for class 2 etc.
    # OUTPUT
    #   target_array: 3D numpy array. '0' for background class, {'1','2',...} for object classes.
    def generate_with_spheres(self, objl, target_array, radius_list):
        """Generates segmentation targets from object list. Here macromolecules are annotated with spheres.
        This method does not require knowledge of the macromolecule shape nor Euler angles in the objl.
        On the other hand, it can be that a network trained with 'sphere targets' is less accurate than with 'shape targets'.

        Args:
            objl (list of dictionaries)
            target_array (3D numpy array): array that initializes the training target. Allows to pass an array already containing annotated structures like membranes.
                index order of array should be [z,y,x]
            radius_list (list of int): contains sphere radii per class (in voxels).
                The radii order in list should correspond to the class label.
                For ex: 1st element of list -> sphere radius for class 1, 2nd element of list -> sphere radius for class 2 etc.

        Returns:
            3D numpy array: Target array, where '0' for background class, {'1','2',...} for object classes.
        """
        Rmax = max(radius_list)
        dim = [2*Rmax, 2*Rmax, 2*Rmax]
        ref_list = []
        for idx in range(len(radius_list)):
            ref_list.append(cm.create_sphere(dim, radius_list[idx]))
        target_array = self.generate_with_shapes(objl, target_array, ref_list)
        return target_array


# TODO: add method for resuming training. It should load existing weights and train_history. So when restarting, the plot curves show prececedent epochs - done by FakET
class Train(core.DeepFinder):
    def __init__(self, Ncl, dim_in):
        core.DeepFinder.__init__(self)
        self.path_out = './'
        self.h5_dset_name = 'dataset' # if training set is stored as .h5 file, specify here in which h5 dataset the arrays are stored

        # Network parameters:
        self.Ncl = Ncl  # Ncl
        self.dim_in = dim_in  # /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
        self.net = models.my_model(self.dim_in, self.Ncl)

        self.label_list = []
        for l in range(self.Ncl): self.label_list.append(l) # for precision_recall_fscore_support
                                                            # (else bug if not all labels exist in batch)

        # Training parameters:
        self.batch_size = 25
        self.epochs = 100
        self.steps_per_epoch = 100
        self.steps_per_valid = 10  # number of samples for validation
        self.optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.loss = losses.tversky_loss

        self.flag_direct_read = 1
        self.flag_batch_bootstrap = 0
        self.Lrnd = 13  # random shifts applied when sampling data- and target-patches (in voxels)

        self.class_weight = None
        self.sample_weights = None  # np array same lenght as objl_train
        
        #save regularly every k'th epoch
        self.save_every = None
        
        self.restart_from_epoch = None

        self.check_attributes()

    def check_attributes(self):
        self.is_positive_int(self.Ncl, 'Ncl')
        self.is_multiple_4_int(self.dim_in, 'dim_in')
        self.is_positive_int(self.batch_size, 'batch_size')
        self.is_positive_int(self.epochs, 'epochs')
        self.is_positive_int(self.steps_per_epoch, 'steps_per_epoch')
        self.is_positive_int(self.steps_per_valid, 'steps_per_valid')
        self.is_int(self.Lrnd, 'Lrnd')

    # This function launches the training procedure. For each epoch, an image is plotted, displaying the progression
    # with different metrics: loss, accuracy, f1-score, recall, precision. Every 10 epochs, the current network weights
    # are saved.
    # INPUTS:
    #   path_data     : a list containing the paths to data files (i.e. tomograms)
    #   path_target   : a list containing the paths to target files (i.e. annotated volumes)
    #   objlist_train : list of dictionaries containing information about annotated objects (e.g. class, position)
    #                   In particular, the tomo_idx should correspond to the index of 'path_data' and 'path_target'.
    #                   See utils/objl.py for more info about object lists.
    #                   During training, these coordinates are used for guiding the patch sampling procedure.
    #   objlist_valid : same as 'objlist_train', but objects contained in this list are not used for training,
    #                   but for validation. It allows to monitor the training and check for over/under-fitting. Ideally,
    #                   the validation objects should originate from different tomograms than training objects.
    # The network is trained on small 3D patches (i.e. sub-volumes), sampled from the larger tomograms (due to memory
    # limitation). The patch sampling is not realized randomly, but is guided by the macromolecule coordinates contained
    # in so-called object lists (objlist).
    # Concerning the loading of the dataset, two options are possible:
    #    flag_direct_read=0: the whole dataset is loaded into memory
    #    flag_direct_read=1: only the patches are loaded into memory, each time a training batch is generated. This is
    #                        usefull when the dataset is too large to load into memory. However, the transfer speed
    #                        between the data server and the GPU host should be high enough, else the procedure becomes
    #                        very slow.
    # TODO: delete flag_direct_read. Launch should detect if direct_read is desired by checking if input data_list and
    #       target_list contain str (path) or numpy array
    def launch(self, path_data, path_target, objlist_train, objlist_valid):
        """This function launches the training procedure. For each epoch, an image is plotted, displaying the progression
        with different metrics: loss, accuracy, f1-score, recall, precision. Every x epochs, the current network weights
        are saved.

        Args:
            path_data (list of string): contains paths to data files (i.e. tomograms)
            path_target (list of string): contains paths to target files (i.e. annotated volumes)
            objlist_train (list of dictionaries): contains information about annotated objects (e.g. class, position)
                In particular, the tomo_idx should correspond to the index of 'path_data' and 'path_target'.
                See utils/objl.py for more info about object lists.
                During training, these coordinates are used for guiding the patch sampling procedure.
            objlist_valid (list of dictionaries): same as 'objlist_train', but objects contained in this list are not
                used for training, but for validation. It allows to monitor the training and check for over/under-fitting.
                Ideally, the validation objects should originate from different tomograms than training objects.

        Note:
            The function saves following files at regular intervals:
                epoch*_epoch.h5: contains current network weights
                train_history.h5: contains arrays with all metrics per training iteration
                train_history.png: plotted metric curves

        """
        self.check_attributes()
        self.check_arguments(path_data, path_target, objlist_train, objlist_valid)
        
        # Logging ###########################################
        logpath = pj(self.path_out, 'logs', 'training')
        logname = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(logpath, exist_ok=True)
        
        # Creating summary in log
        summary = {
            "training_tomograms" : path_data,
            "num_epochs" : self.epochs,
            "seed": self.seed,
            "Nclass": self.Ncl,
            "dim_in": self.dim_in,
            "batch_size": self.batch_size,
            "Nvalid": self.Nvalid,
            "flag_direct_read": self.flag_direct_read,
            "flag_batch_bootstrap": self.flag_batch_bootstrap,
            "Lrnd": self.Lrnd,
            "class_weights": self.class_weights,
            }
        with open(pj(logpath, f'{logname}.json'), 'w') as fsum:
            json.dump(summary, fsum, indent=4)

        # Redirect stdout to log
        outlog = pj(logpath, f'{logname}.out')
        fout = open(outlog, 'a+')
        origstdout = sys.stdout
        sys.stdout = fout

        # Redirect stderr to log
        errlog = pj(logpath, f'{logname}.err')
        ferr = open(errlog, 'a+')
        origstderr = sys.stderr
        sys.stderr = ferr
        #####################################################
        
        if self.restart_from_epoch is None:
            self.display('Compiling the network ...')
            # Build network (not in constructor, else not possible to init model with weights from previous train round):
            self.net.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
            self.restart_from_epoch = 1
            # Declare lists for storing training statistics:
            hist_loss_train = []
            hist_acc_train  = []
            hist_loss_valid = []
            hist_acc_valid  = []
            hist_f1         = []
            hist_recall     = []
            hist_precision  = []
            process_time    = []
        else:
            self.display(f'Restarting training from epoch {self.restart_from_epoch}.')
            # Load lists with previous history of training statistics:
            hist = core.read_history(pj(logpath, 'train_history.h5'))
            hist_loss_train = hist['loss'].tolist()
            hist_acc_train  = hist['acc'].tolist()
            hist_loss_valid = hist['val_loss'].tolist()
            hist_acc_valid  = hist['val_acc'].tolist()
            hist_f1         = hist['val_f1'].tolist()
            hist_recall     = hist['val_recall'].tolist()
            hist_precision  = hist['val_precision'].tolist()
            process_time    = []
            
        # Load whole dataset:
        if self.flag_direct_read == False:
            self.display('Loading dataset ...')
            data_list, target_list = core.load_dataset(path_data, path_target, self.h5_dset_name)

        self.display('Launch training ...')     

        # Training loop:
        best_epoch, best_f1 = 1, 0.  # Keep info of best epoch
        for e in range(self.restart_from_epoch - 1, self.epochs):
            # TRAINING:
            start = time.time()
            list_loss_train = []
            list_acc_train = []

            # FakET change how one epoch is defined, before, one epoch was just a
            # specified number of training steps, but now one epoch iterates over
            # all training data in random order such that each instance is seen at
            # most once. Seed for initialization of the network itself does not 
            # influence ordering of batches,
            
            # Random shuffle batch indices
            rng = np.random.default_rng(12345+e)   
            idxs = list(rng.permutation(range(len(objlist_train))))
            n_idxs = len(idxs)

            # Iterate over all data instead of specified number of steps
            #n_iterations = self.steps_per_epoch
            n_iterations = int(len(objlist_train) / self.batch_size)

            # Timing data generation
            generating_data_time = 0
            
            for it in range(n_iterations):
                t0 = time.time()
                if (it+1)*self.batch_size < n_idxs:
                    batch_idxs = idxs[it*self.batch_size: (it+1)*self.batch_size]
                else:  # Last batch might have less samples
                    batch_idxs = idxs[it*self.batch_size:]
                    batch_idxs = batch_idxs + idxs[:self.batch_size - len(batch_idxs)]


                if self.flag_direct_read:
                    batch_data, batch_target = self.generate_batch_direct_read(path_data, path_target, self.batch_size, objlist_train, batch_idxs=batch_idxs)
                else:
                    batch_data, batch_target, idx_list = self.generate_batch_from_array(data_list, target_list, self.batch_size, objlist_train, batch_idxs=batch_idxs)
                    
                sample_weight = None if self.sample_weights is None else self.sample_weights[idx_list]
                    
                generating_data_time += time.time() - t0
                
                loss_train = self.net.train_on_batch(batch_data, batch_target,
                                                     class_weight=self.class_weight,
                                                     sample_weight=sample_weight)

                self.display('epoch %d/%d - it %d/%d - loss: %0.3f - acc: %0.3f' % (e + 1, self.epochs, it + 1, n_iterations, loss_train[0], loss_train[1]))
                list_loss_train.append(loss_train[0])
                list_acc_train.append(loss_train[1])
                
                # Write stdout to file immediately
                fout.flush()
                
            hist_loss_train.append(list_loss_train)
            hist_acc_train.append(list_acc_train)

            
            valid_start = time.time()
            # VALIDATION (compute statistics to monitor training):
            list_loss_valid = []
            list_acc_valid  = []
            list_f1         = []
            list_recall     = []
            list_precision  = []
            for it in range(self.steps_per_valid):
                if self.flag_direct_read:
                    batch_data_valid, batch_target_valid = self.generate_batch_direct_read(path_data, path_target, self.batch_size, objlist_valid)
                else:
                    batch_data_valid, batch_target_valid, idx_list = self.generate_batch_from_array(data_list, target_list, self.batch_size, objlist_valid)
                
                
                # Faket fix of memory leak (when numpy array is passed, creates new graph)
                # Read more here: https://medium.com/dive-into-ml-ai/dealing-with-memory-leak-issue-in-keras-model-training-e703907a6501
                batch_data_valid = tf.convert_to_tensor(batch_data_valid)  
                batch_pred = self.net.predict_on_batch(batch_data_valid)
                loss_val = self.net.evaluate(batch_data_valid, batch_target_valid, verbose=0, batch_size=len(batch_data_valid)) # TODO replace by loss() to reduce computation
                
                # batch_pred = self.net.predict(batch_data_valid)
                #loss_val = K.eval(losses.tversky_loss(K.constant(batch_target_valid), K.constant(batch_pred)))
                scores = precision_recall_fscore_support(batch_target_valid.argmax(axis=-1).flatten(), 
                                                         batch_pred.argmax(axis=-1).flatten(), average=None, 
                                                         labels=self.label_list)

                list_loss_valid.append(loss_val[0])
                list_acc_valid.append(loss_val[1])
                list_f1.append(scores[2])
                list_recall.append(scores[1])
                list_precision.append(scores[0])
            
            # Update info on the best epoch
            if np.mean(list_f1) > best_f1:
                best_f1 = np.mean(list_f1)
                best_epoch = e + 1
            
            hist_loss_valid.append(list_loss_valid)
            hist_acc_valid.append(list_acc_valid)
            hist_f1.append(list_f1)
            hist_recall.append(list_recall)
            hist_precision.append(list_precision)

            end = time.time()
            process_time.append(end - start)
            self.display('-------------------------------------------------------------')
            self.display((
                f'EPOCH {e + 1}/{self.epochs} - valid loss: {loss_val[0]:.3f} - '
                f'valid acc: {loss_val[1]:.3f} - {end - start:.2f}sec '
                f'(from which {generating_data_time:.2f}sec generating data & '
                f'{end - valid_start:.2f}sec validating)'
            ))
            
            # Save and plot training history:
            history = {'loss': hist_loss_train, 'acc': hist_acc_train, 'val_loss': hist_loss_valid,
                       'val_acc': hist_acc_valid, 'val_f1': hist_f1, 'val_recall': hist_recall,
                       'val_precision': hist_precision}
            
            
            core.save_history(history, pj(logpath, 'train_history.h5'))
            core.plot_history(history, pj(logpath, 'train_history.png'))

            self.display('=============================================================')
            
            if self.save_every is not None:
                if (e + 1) % self.save_every == 0:  # save weights every epochs
                    self.net.save(pj(self.path_out, f'epoch{e + 1:03d}_weights.h5'))
                    
            # Housekeeping
            gc.collect()
            K.clear_session()

        self.display(f"Model took {np.sum(process_time):.2f} seconds to train since last restart.")
        self.display(f"Best model according to val_f1 is {best_epoch:03d} with score: {best_f1:.4f}.")
        
        # If the last epoch was not saved at `save_every`, save it now
        save_final = self.save_every or (e + 1)
        if (e + 1) % save_final == 0:
            self.net.save(pj(self.path_out, f'epoch{e + 1:03d}_weights.h5'))
        
        # Close log files & remove empty log files if any
        fout.close()
        ferr.close()
        for log in [outlog, errlog]:
            if os.path.isfile(log) and os.path.getsize(log) == 0:
                os.remove(log)
                
        # Redirecting back
        sys.stdout = origstdout
        sys.stderr = origstderr


    def check_arguments(self, path_data, path_target, objlist_train, objlist_valid):
        self.is_list(path_data, 'path_data')
        self.is_list(path_target, 'path_target')
        self.are_lists_same_length([path_data, path_target], ['data_list', 'target_list'])
        self.is_list(objlist_train, 'objlist_train')
        self.is_list(objlist_valid, 'objlist_valid')

    # Generates batches for training and validation. In this version, the dataset is not loaded into memory. Only the
    # current batch content is loaded into memory, which is useful when memory is limited.
    # Is called when self.flag_direct_read=True
    # !! For now only works for h5 files !!
    # !! FakET - now also mrc files are supported !!
    # Batches are generated as follows:
    #   - The positions at which patches are sampled are determined by the coordinates contained in the object list.
    #   - Two data augmentation techniques are applied:
    #       .. To gain invariance to translations, small random shifts are added to the positions.
    #       .. 180 degree rotation around tilt axis (this way missing wedge orientation remains the same).
    #   - Also, bootstrap (i.e. re-sampling) can be applied so that we have an equal amount of each macromolecule in
    #     each batch. This is useful when a class is under-represented. Is applied when self.flag_batch_bootstrap=True
    # INPUTS:
    #   path_data: list of strings '/path/to/tomo.h5'
    #   path_target: list of strings '/path/to/target.h5'
    #   batch_size: int
    #   objlist: list of dictionnaries
    # OUTPUT:
    #   batch_data: numpy array [batch_idx, z, y, x, channel] in our case only 1 channel
    #   batch_target: numpy array [batch_idx, z, y, x, class_idx] is one-hot encoded
    def generate_batch_direct_read(self, path_data, path_target, batch_size, objlist=None, batch_idxs=None):
        p_in = np.int(np.floor(self.dim_in / 2))

        batch_data = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, 1))
        batch_target = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, self.Ncl))

        # The batch is generated by randomly sampling data patches.
        if self.flag_batch_bootstrap:  # choose from bootstrapped objlist
            pool = core.get_bootstrap_idx(objlist, Nbs=batch_size)
        else:  # choose from whole objlist
            pool = range(0, len(objlist))

        for i in range(batch_size):
            # Faket fixing seed to get the same results when using 
            # `generate_batch_direct_read` and `generate_batch_from_array`
            rng = np.random.default_rng(12345+i)
                
            # If batch indices are given
            index = 0
            if batch_idxs is not None:
                index = batch_idxs[i]
            # Choose random object in training set:
            else: 
                index = rng.choice(pool)

            # FakET support for loading mrc files as memmory maps for direct read
            tomoID = int(objlist[index]['tomo_idx'])
            data_file_path = path_data[tomoID]
            target_file_path = path_target[tomoID]
        
            if data_file_path.endswith('.mrc') and target_file_path.endswith('.mrc'):
                import mrcfile
                data_file = mrcfile.mmap(data_file_path, mode='r')
                data = data_file.data
                target_file = mrcfile.mmap(target_file_path, mode='r')
                target = target_file.data
            elif data_file_path.endswith('.h5') and target_file_path.endswith('.h5'):
                data_file = h5py.File(data_file_path, 'r')
                data = data_file['dataset']
                target_file = h5py.File(target_file_path, 'r')
                target = target_file['dataset']
            else:
                raise ValueError('With direct read, only .mrc and .h5 files are supported.')
            
            tomodim = data.shape
            x, y, z = core.get_patch_position(tomodim, p_in, objlist[index], self.Lrnd)
            patch_data = data[z-p_in:z+p_in, y-p_in:y+p_in, x-p_in:x+p_in]
            patch_target = target[z-p_in:z+p_in, y-p_in:y+p_in, x-p_in:x+p_in]
            data_file.close()
            target_file.close()
            # FakET changes end -------------------------------------------------------

            # Original code by DeepFinder (which supported only h5 files)
            # h5file = h5py.File(path_data[tomoID], 'r')
            # tomodim = h5file['dataset'].shape  # get tomo dimensions without loading the array
            # h5file.close()
            # x, y, z = core.get_patch_position(tomodim, p_in, objlist[index], self.Lrnd)
            # # Load data and target patches:
            # h5file = h5py.File(path_data[tomoID], 'r')
            # patch_data = h5file['dataset'][z-p_in:z+p_in, y-p_in:y+p_in, x-p_in:x+p_in]
            # h5file.close()
            # h5file = h5py.File(path_target[tomoID], 'r')
            # patch_target = h5file['dataset'][z-p_in:z+p_in, y-p_in:y+p_in, x-p_in:x+p_in]
            # h5file.close()

            # Process the patches in order to be used by network:
            patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)  # normalize
            patch_target_onehot = to_categorical(patch_target, self.Ncl)

            # Store into batch array:
            batch_data[i, :, :, :, 0] = patch_data
            batch_target[i] = patch_target_onehot

            # Data augmentation (180degree rotation around tilt axis):
            if rng.uniform() < 0.5:
                batch_data[i] = np.rot90(batch_data[i], k=2, axes=(0, 2))
                batch_target[i] = np.rot90(batch_target[i], k=2, axes=(0, 2))

        return batch_data, batch_target

    # Generates batches for training and validation. In this version, the whole dataset has already been loaded into
    # memory, and batch is sampled from there. Apart from that does the same as above.
    # Is called when self.flag_direct_read=False
    # INPUTS:
    #   data: list of numpy arrays
    #   target: list of numpy arrays
    #   batch_size: int
    #   objlist: list of dictionnaries
    # OUTPUT:
    #   batch_data: numpy array [batch_idx, z, y, x, channel] in our case only 1 channel
    #   batch_target: numpy array [batch_idx, z, y, x, class_idx] is one-hot encoded
    def generate_batch_from_array(self, data, target, batch_size, objlist=None, batch_idxs=None):
        p_in = np.int(np.floor(self.dim_in / 2))

        batch_data = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, 1))
        batch_target = np.zeros((batch_size, self.dim_in, self.dim_in, self.dim_in, self.Ncl))

        # The batch is generated by randomly sampling data patches.
        if self.flag_batch_bootstrap:  # choose from bootstrapped objlist
            pool = core.get_bootstrap_idx(objlist, Nbs=batch_size)
        else:  # choose from whole objlist
            pool = range(0, len(objlist))

        idx_list = []
        for i in range(batch_size):
            # choose random sample in training set:
            rng = np.random.default_rng(12345+i)  
            
            # FakET - Following lines copied from generate_batch_direct_read
            # because the two functions did not behave exactly the same
            # way which influenced the reproducibility of the results
            # If batch indices are given
            index = 0
            if batch_idxs is not None:
                index = batch_idxs[i]
            # Choose random object in training set:
            else: 
                index = rng.choice(pool)

            idx_list.append(index)

            tomoID = int(objlist[index]['tomo_idx'])
            tomodim = data[tomoID].shape

            sample_data = data[tomoID]
            sample_target = target[tomoID]

            # Get patch position:
            x, y, z = core.get_patch_position(tomodim, p_in, objlist[index], self.Lrnd)

            # Get patch:
            patch_data   = sample_data[  z-p_in:z+p_in, y-p_in:y+p_in, x-p_in:x+p_in]
            patch_target = sample_target[z-p_in:z+p_in, y-p_in:y+p_in, x-p_in:x+p_in]

            # Process the patches in order to be used by network:
            patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)  # normalize
            patch_target_onehot = to_categorical(patch_target, self.Ncl)

            # Store into batch array:
            batch_data[i, :, :, :, 0] = patch_data
            batch_target[i] = patch_target_onehot

            # Data augmentation (180degree rotation around tilt axis):
            if rng.uniform() < 0.5:
                batch_data[i] = np.rot90(batch_data[i], k=2, axes=(0, 2))
                batch_target[i] = np.rot90(batch_target[i], k=2, axes=(0, 2))

        return batch_data, batch_target, idx_list

