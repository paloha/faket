# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

import sys
from lxml import etree

class ParamsGenTarget():
    def __init__(self):
        self.path_objl = str()
        self.path_initial_vol = str()
        self.tomo_size = (int(), int(), int()) # (dimZ,dimY,dimX)
        self.strategy = str()
        self.radius_list = [int()]
        self.path_mask_list = [str()]
        self.path_target = str()

    def check(self):
        if type(self.path_objl)!=str:
            print('DeepFinder message: path_objl needs to be of type str.')
        if type(self.path_initial_vol)!=str:
            print('DeepFinder message: path_initial_vol needs to be of type str.')
        if type(self.strategy)!=str:
            print('DeepFinder message: strategy needs to be of type str.')
        if self.strategy!='spheres' and self.strategy!='shapes':
            print('DeepFinder message: strategy can only be "spheres" or "shapes".')
        for r in self.radius_list:
            if type(r)!=int:
                print('DeepFinder message: radius_list must contain only integers.')
        for p in self.path_mask_list:
            if type(p)!=str:
                print('DeepFinder message: path_mask_list must contain only strings.')

    def write(self, filename):
        root = etree.Element('paramsGenerateTarget')

        p = etree.SubElement(root, 'path_objl')
        p.set('path', str(self.path_objl))

        p = etree.SubElement(root, 'path_initial_vol')
        p.set('path', str(self.path_initial_vol))

        p = etree.SubElement(root, 'tomo_size')
        pp = etree.SubElement(p, 'X')
        pp.set('size', str(self.tomo_size[2]))
        pp = etree.SubElement(p, 'Y')
        pp.set('size', str(self.tomo_size[1]))
        pp = etree.SubElement(p, 'Z')
        pp.set('size', str(self.tomo_size[0]))

        p = etree.SubElement(root, 'strategy')
        p.set('strategy', str(self.strategy))

        p = etree.SubElement(root, 'radius_list')
        for idx in range(len(self.radius_list)):
            pp = etree.SubElement(p, 'class' + str(idx + 1))
            pp.set('radius', str(self.radius_list[idx]))

        p = etree.SubElement(root, 'path_mask_list')
        for idx in range(len(self.path_mask_list)):
            pp = etree.SubElement(p, 'class'+str(idx + 1))
            pp.set('path', str(self.path_mask_list[idx]))

        p = etree.SubElement(root, 'path_target')
        p.set('path', str(self.path_target))

        tree = etree.ElementTree(root)
        tree.write(filename, pretty_print=True)

    def read(self, filename):
        tree = etree.parse(filename)
        root = tree.getroot()

        if root.tag == 'paramsGenerateTarget':
            self.path_objl = root.find('path_objl').get('path')
            self.path_initial_vol = root.find('path_initial_vol').get('path')

            x = int( root.find('tomo_size').find('X').get('size') )
            y = int( root.find('tomo_size').find('Y').get('size') )
            z = int( root.find('tomo_size').find('Z').get('size') )
            self.tomo_size = (z, y, x)

            self.strategy = root.find('strategy').get('strategy')

            self.radius_list = []
            for idx in range(len(root.find('radius_list'))):
                radius = root.find('radius_list').find('class'+str(idx+1)).get('radius')
                self.radius_list.append( int(radius) )

            self.path_mask_list = []
            for idx in range(len(root.find('path_mask_list'))):
                path = root.find('path_mask_list').find('class' + str(idx + 1)).get('path')
                self.path_mask_list.append(path)

            self.path_target = root.find('path_target').get('path')

        else:
            print('DeepFinder message: wrong params file.')
            sys.exit()

    def display(self):
        print('path_objl       : ' + self.path_objl)
        print('path_initial_vol: ' + self.path_initial_vol)
        print('tomo size       : ' + str(self.tomo_size))  # (dimZ,dimY,dimX)
        print('strategy        : ' + self.strategy)
        print('radius list     : ' + str(self.radius_list))
        print('path_mask_list  : ' + str(self.path_mask_list))
        print('path_target     : ' + self.path_target)


class ParamsTrain():
    def __init__(self):
        self.path_out = str()
        self.path_tomo = [str()]
        self.path_target = [str()]
        self.path_objl_train = str()
        self.path_objl_valid = str()
        self.Ncl = int()
        self.psize = int()
        self.bsize = int()
        self.nepochs = int()
        self.steps_per_e = int()
        self.steps_per_v = int()
        self.flag_direct_read = bool()
        self.flag_bootstrap = bool()
        self.rnd_shift = int()

    def write(self, filename):
        root = etree.Element('paramsTrain')

        p = etree.SubElement(root, 'path_out')
        p.set('path', str(self.path_out))

        p = etree.SubElement(root, 'path_tomo')
        for idx in range(len(self.path_tomo)):
            pp = etree.SubElement(p, 'tomo' + str(idx))
            pp.set('path', str(self.path_tomo[idx]))

        p = etree.SubElement(root, 'path_target')
        for idx in range(len(self.path_target)):
            pp = etree.SubElement(p, 'target' + str(idx))
            pp.set('path', str(self.path_target[idx]))


        p = etree.SubElement(root, 'path_objl_train')
        p.set('path', str(self.path_objl_train))

        p = etree.SubElement(root, 'path_objl_valid')
        p.set('path', str(self.path_objl_valid))

        p = etree.SubElement(root, 'number_of_classes')
        p.set('n', str(self.Ncl))

        p = etree.SubElement(root, 'patch_size')
        p.set('n', str(self.psize))

        p = etree.SubElement(root, 'batch_size')
        p.set('n', str(self.bsize))

        p = etree.SubElement(root, 'number_of_epochs')
        p.set('n', str(self.nepochs))

        p = etree.SubElement(root, 'steps_per_epoch')
        p.set('n', str(self.steps_per_e))

        p = etree.SubElement(root, 'steps_per_validation')
        p.set('n', str(self.steps_per_v))

        p = etree.SubElement(root, 'flag_direct_read')
        p.set('flag', str(self.flag_direct_read))

        p = etree.SubElement(root, 'flag_bootstrap')
        p.set('flag', str(self.flag_bootstrap))

        p = etree.SubElement(root, 'random_shift')
        p.set('shift', str(self.rnd_shift))

        tree = etree.ElementTree(root)
        tree.write(filename, pretty_print=True)

    def read(self, filename):
        tree = etree.parse(filename)
        root = tree.getroot()

        if root.tag=='paramsTrain':
            self.path_out = root.find('path_out').get('path')

            self.path_tomo = []
            for idx in range(len(root.find('path_tomo'))):
                path = root.find('path_tomo').find('tomo' + str(idx)).get('path')
                self.path_tomo.append(path)

            self.path_target = []
            for idx in range(len(root.find('path_target'))):
                path = root.find('path_target').find('target' + str(idx)).get('path')
                self.path_target.append(path)

            self.path_objl_train = root.find('path_objl_train').get('path')
            self.path_objl_valid = root.find('path_objl_valid').get('path')
            self.Ncl = int( root.find('number_of_classes').get('n') )
            self.psize = int( root.find('patch_size').get('n') )
            self.bsize = int( root.find('batch_size').get('n') )
            self.nepochs = int( root.find('number_of_epochs').get('n') )
            self.steps_per_e = int( root.find('steps_per_epoch').get('n') )
            self.steps_per_v = int( root.find('steps_per_validation').get('n') )

            flag = root.find('flag_direct_read').get('flag')
            if flag=='True': self.flag_direct_read = True
            elif flag=='False': self.flag_direct_read = False

            flag = root.find('flag_bootstrap').get('flag')
            if flag=='True': self.flag_bootstrap = True
            elif flag=='False': self.flag_bootstrap = False

            self.rnd_shift = int( root.find('random_shift').get('shift') )
        else:
            print('DeepFinder message: wrong params file.')
            sys.exit()


class ParamsSegment():
    def __init__(self):
        self.Ncl = int()
        self.psize = int()
        self.path_weights = str()
        self.path_tomo = str()
        self.path_lmap = str()

    def write(self, filename):
        root = etree.Element('paramsSegment')

        p = etree.SubElement(root, 'number_of_classes')
        p.set('n', str(self.Ncl))

        p = etree.SubElement(root, 'patch_size')
        p.set('n', str(self.psize))

        p = etree.SubElement(root, 'path_net_weights')
        p.set('path', str(self.path_weights))

        p = etree.SubElement(root, 'path_tomo')
        p.set('path', str(self.path_tomo))

        p = etree.SubElement(root, 'path_lmap')
        p.set('path', str(self.path_lmap))

        tree = etree.ElementTree(root)
        tree.write(filename, pretty_print=True)

    def read(self, filename):
        tree = etree.parse(filename)
        root = tree.getroot()

        if root.tag == 'paramsSegment':
            self.Ncl = int( root.find('number_of_classes').get('n') )
            self.psize = int( root.find('patch_size').get('n') )
            self.path_weights = root.find('path_net_weights').get('path')
            self.path_tomo = root.find('path_tomo').get('path')
            self.path_lmap = root.find('path_lmap').get('path')
        else:
            print('DeepFinder message: wrong params file.')
            sys.exit()


class ParamsCluster():
    def __init__(self):
        self.path_lmap = str()
        self.cradius = int()
        #self.csize_thr = None
        self.path_objl = str()

    def write(self, filename):
        root = etree.Element('paramsCluster')

        p = etree.SubElement(root, 'path_label_map')
        p.set('path', str(self.path_lmap))

        p = etree.SubElement(root, 'clustering_radius')
        p.set('radius', str(self.cradius))

        p = etree.SubElement(root, 'path_objl')
        p.set('path', str(self.path_objl))

        tree = etree.ElementTree(root)
        tree.write(filename, pretty_print=True)

    def read(self, filename):
        tree = etree.parse(filename)
        root = tree.getroot()

        if root.tag == 'paramsCluster':
            self.path_lmap = root.find('path_label_map').get('path')
            self.cradius = int(root.find('clustering_radius').get('radius'))
            self.path_objl = root.find('path_objl').get('path')
        else:
            print('DeepFinder message: wrong params file.')
            sys.exit()