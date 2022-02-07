

import utils.objl as ol


label_dict = {
                '4V94' : 1,
                '4CR2' : 2,
                '1QVR' : 3,
                '1BXN' : 4,
                '3CF3' : 5,
                '1U6G' : 6,
                '3D2F' : 7,
                '2CG9' : 8,
                '3H84' : 9,
                '3GL1' : 10,
                '3QM1' : 11,
                '1S3X' : 12,
                '5MRC' : 13,
                'vesicle' : 14,
                'fiducial' : 15}

def read_txt(filename, idx):
    tomo_idx = idx 

    objlOUT = []
    with open(str(filename), 'rU') as f:
        for line in f:
            lbl, x, y, z, *_ = line.rstrip('\n').split()
            ol.add_obj(objlOUT, tomo_idx=tomo_idx ,label=label_dict[lbl], coord=(float(x), float(y), float(z)))
    return objlOUT



#filename = 'objl_valid_2021.xml'
#filename = 'objl_training_2021.xml'
#filename = 'objl_training_2021_double.xml'


#paths = ['../../../faket/data/shrec2021_contest_dataset/model_8/particle_locations.txt']

def create_objl(paths, tomo_id=None):
    objl = []
    for idx, path in enumerate(paths):
        if tomo_id is not None:
            objl =objl + read_txt(path, tomo_id)
        else:
            objl =objl + read_txt(path, idx)
    return objl



#ol.write(objl, filename)


