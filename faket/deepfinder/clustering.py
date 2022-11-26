# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

import numpy as np
import time

from sklearn.cluster import MeanShift

from .utils import objl as ol
from .utils import core


class Cluster(core.DeepFinder):
    def __init__(self, clustRadius):
        core.DeepFinder.__init__(self)
        self.clustRadius = clustRadius
        self.sizeThr = 1 # TODO: delete

        self.check_attributes()

    def check_attributes(self):
        self.is_positive_int(self.clustRadius, 'clustRadius')


    # This function analyzes the segmented tomograms (i.e. labelmap), identifies individual macromolecules and outputs their coordinates. This is achieved with a clustering algorithm (meanshift).
    # INPUTS:
    #   labelmap: segmented tomogram (3D numpy array)
    #   sizeThr : cluster size (i.e. macromolecule size) (in voxels), under which a detected object is considered a false positive and is discarded
    #   clustRadius: parameter for clustering algorithm. Corresponds to average object radius (in voxels)
    # OUTPUT:
    #   objlist: a xml structure containing infos about detected objects: coordinates, class label and cluster size
    def launch(self, labelmap, n_jobs=1):
        """This function analyzes the segmented tomograms (i.e. labelmap), identifies individual macromolecules and outputs
        their coordinates. This is achieved with a clustering algorithm (meanshift).

        Args:
            labelmap (3D numpy array): segmented tomogram
            clustRadius (int): parameter for clustering algorithm. Corresponds to average object radius (in voxels)

        Returns:
            list of dict: the object list with coordinates and class labels of identified macromolecules
        """
        self.check_arguments(labelmap)

        Nclass = len(np.unique(labelmap)) - 1  # object classes only (background class not considered)

        objvoxels = np.nonzero(labelmap > 0)
        objvoxels = np.array(objvoxels).T  # convert to numpy array and transpose

        self.display(f'Launch clustering of {len(objvoxels)} objvoxels in {n_jobs} parallel jobs...')
        start = time.time()
        clusters = MeanShift(bandwidth=self.clustRadius, bin_seeding=True, n_jobs=n_jobs).fit(objvoxels)
        end = time.time()
        self.display("Clustering took %0.2f seconds" % (end - start))

        Nclust = clusters.cluster_centers_.shape[0]

        self.display('Analyzing clusters ...')
        objlist = []
        labelcount = np.zeros((Nclass,))
        for c in range(Nclust):
            clustMemberIndex = np.nonzero(clusters.labels_ == c)

            # Get cluster size and position:
            clustSize = np.size(clustMemberIndex)
            centroid = clusters.cluster_centers_[c]

            # Attribute a macromolecule class to cluster:
            clustMember = []
            for m in range(clustSize):  # get labels of cluster members
                clustMemberCoords = objvoxels[clustMemberIndex[0][m], :]
                clustMember.append(labelmap[clustMemberCoords[0], clustMemberCoords[1], clustMemberCoords[2]])

            for l in range(Nclass):  # get most present label in cluster
                labelcount[l] = np.size(np.nonzero(np.array(clustMember) == l + 1))
            winninglabel = np.argmax(labelcount) + 1

            objlist = ol.add_obj(objlist, label=winninglabel, coord=centroid, cluster_size=clustSize)

        self.display('Finished !')
        self.display_result(objlist)
        return objlist

    def check_arguments(self, labelmap):
        self.is_3D_nparray(labelmap, 'labelmap')

    def display_result(self, objlist):
        self.display('----------------------------------------')
        self.display('A total of ' + str(len(objlist)) + ' objects has been found.')
        lbl_list = ol.get_labels(objlist)
        for lbl in lbl_list:
            objl_class = ol.get_class(objlist, lbl)
            self.display('Class ' + str(lbl) + ': ' + str(len(objl_class)) + ' objects')
