"""Neural style transfer (https://arxiv.org/abs/1508.06576) in PyTorch."""

import os
import argparse
import atexit
from dataclasses import asdict
import io
import json
from pathlib import Path
import platform
import sys
import numpy as np
from PIL import Image, ImageCms
from tifffile import TIFF, TiffWriter
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import numpy as np
import random
import time

# Changed by faket to fix imports in current project structure
srgb_profile = (Path(__file__).resolve().parent / 'sRGB Profile.icc').read_bytes()
from .style_transfer import STIterate, StyleTransfer
from ..data import load_mrc, save_mrc, normalize, match_mean_std


def prof_to_prof(image, src_prof, dst_prof, **kwargs):
    src_prof = io.BytesIO(src_prof)
    dst_prof = io.BytesIO(dst_prof)
    return ImageCms.profileToProfile(image, src_prof, dst_prof, **kwargs)


def load_image(path, proof_prof=None):
    src_prof = dst_prof = srgb_profile
    try:
        image = Image.open(path)
        if 'icc_profile' in image.info:
            src_prof = image.info['icc_profile']
        else:
            image = image.convert('RGB')
        if proof_prof is None:
            if src_prof == dst_prof:
                return image.convert('RGB')
            return prof_to_prof(image, src_prof, dst_prof, outputMode='RGB')
        proof_prof = Path(proof_prof).read_bytes()
        cmyk = prof_to_prof(image, src_prof, proof_prof, outputMode='CMYK')
        return prof_to_prof(cmyk, proof_prof, dst_prof, outputMode='RGB')
    except OSError as err:
        print_error(err)
        sys.exit(1)


def save_pil(path, image):
    try:
        kwargs = {'icc_profile': srgb_profile}
        if path.suffix.lower() in {'.jpg', '.jpeg'}:
            kwargs['quality'] = 95
            kwargs['subsampling'] = 0
        elif path.suffix.lower() == '.webp':
            kwargs['quality'] = 95
        image.save(path, **kwargs)
    except (OSError, ValueError) as err:
        print_error(err)
        sys.exit(1)


def save_tiff(path, image):
    tag = ('InterColorProfile', TIFF.DATATYPES.BYTE, len(srgb_profile), srgb_profile, False)
    try:
        with TiffWriter(path) as writer:
            writer.save(image, photometric='rgb', resolution=(72, 72), extratags=[tag])
    except OSError as err:
        print_error(err)
        sys.exit(1)


def save_image(path, image):
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    # tqdm.write(f'Writing image to {path}.')
    if isinstance(image, Image.Image):
        save_pil(path, image)
    elif isinstance(image, np.ndarray) and path.suffix.lower() in {'.tif', '.tiff'}:
        save_tiff(path, image)
    else:
        raise ValueError('Unsupported combination of image type and extension')


def get_safe_scale(w, h, dim):
    """Given a w x h content image and that a dim x dim square does not
    exceed GPU memory, compute a safe end_scale for that content image."""
    return int(pow(w / h if w > h else h / w, 1/2) * dim)


def setup_exceptions():
    try:
        from IPython.core.ultratb import FormattedTB
        sys.excepthook = FormattedTB(mode='Plain', color_scheme='Neutral')
    except ImportError:
        pass


def fix_start_method():
    if platform.system() == 'Darwin':
        mp.set_start_method('spawn')


def print_error(err):
    print('\033[31m{}:\033[0m {}'.format(type(err).__name__, err), file=sys.stderr)


class Callback:
    def __init__(self, st, args, image_type='pil', web_interface=None, seq_i=None, seq_len=None):
        self.st = st
        self.args = args
        self.image_type = image_type
        self.web_interface = web_interface
        self.iterates = []
        self.progress = None
        self.seq_i = seq_i
        self.seq_len = seq_len

    def __call__(self, iterate):
        self.iterates.append(asdict(iterate))
        if iterate.i == 1:
            self.progress = tqdm(total=iterate.i_max, dynamic_ncols=True)
        out_fpath = self.args.output.format(self.seq_i, iterate.w, iterate.h, iterate.i)
        if iterate.i % self.args.save_every == 0:
            msg = f'Image: {self.seq_i}/{self.seq_len} '
            msg += 'Size: {}x{}, iteration: {}, loss: {:g}'
            tqdm.write(msg.format(iterate.w, iterate.h, iterate.i, iterate.loss))
            save_image(out_fpath, self.st.get_image(self.image_type))
        self.progress.update()
        if iterate.i == iterate.i_max:
            self.progress.close()

    def close(self):
        if self.progress is not None:
            self.progress.close()

    def get_trace(self):
        return {'args': self.args.__dict__, 'iterates': self.iterates}


def main():
    setup_exceptions()
    fix_start_method()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def arg_info(arg):
        defaults = StyleTransfer.stylize.__kwdefaults__
        default_types = StyleTransfer.stylize.__annotations__
        return {'default': defaults[arg], 'type': default_types[arg]}

    p.add_argument('content', type=str, help='the content image')
    p.add_argument('styles', type=str, nargs='+', metavar='style', help='the style images')
    p.add_argument('--output', '-o', type=str, default='out.png',
                   help='the output image. Add "{:02d}" to keep save_every img')
    p.add_argument('--style-weights', '-sw', type=float, nargs='+', default=None,
                   metavar='STYLE_WEIGHT', help='the relative weights for each style image')
    p.add_argument('--devices', type=str, default=[], nargs='+',
                   help='the device names to use (omit for auto)')
    p.add_argument('--random-seed', '-r', type=int, default=0,
                   help='the random seed for optimizer & optionally the model initialization')
    p.add_argument('--content-weight', '-cw', **arg_info('content_weight'),
                   help='the content weight')
    p.add_argument('--tv-weight', '-tw', **arg_info('tv_weight'),
                   help='the smoothing weight')
    p.add_argument('--min-scale', '-ms', **arg_info('min_scale'),
                   help='the minimum scale (max image dim), in pixels')
    p.add_argument('--end-scale', '-s', type=str, default='512',
                   help='the final scale (max image dim), in pixels')
    p.add_argument('--iterations', '-i', **arg_info('iterations'),
                   help='the number of iterations per scale')
    p.add_argument('--initial-iterations', '-ii', **arg_info('initial_iterations'),
                   help='the number of iterations on the first scale')
    p.add_argument('--save-every', type=int, default=50,
                   help='save the image every SAVE_EVERY iterations')
    p.add_argument('--step-size', '-ss', **arg_info('step_size'),
                   help='the step size (learning rate)')
    p.add_argument('--avg-decay', '-ad', **arg_info('avg_decay'),
                   help='the EMA decay rate for iterate averaging')
    p.add_argument('--init', type=str, default='content',
                   help="the initial image: mrc file or {'content', 'gray', 'uniform', 'style_mean'}")
    p.add_argument('--style-scale-fac', **arg_info('style_scale_fac'),
                   help='the relative scale of the style to the content')
    p.add_argument('--style-size', **arg_info('style_size'),
                   help='the fixed scale of the style at different content scales')
    p.add_argument('--pooling', type=str, default='max', choices=['max', 'average', 'l2'],
                   help='the model\'s pooling mode')
    p.add_argument('--proof', type=str, default=None,
                   help='the ICC color profile (CMYK) for soft proofing the content and styles')
    p.add_argument('--web', default=False, action='store_true', help='enable the web interface')
    p.add_argument('--host', type=str, default='0.0.0.0',
                   help='the host the web interface binds to')
    p.add_argument('--port', type=int, default=8080,
                   help='the port the web interface binds to')
    p.add_argument('--browser', type=str, default='', nargs='?',
                   help='open a web browser (specify the browser if not system default)')
    p.add_argument('--seq_start', type=int, default=0,
                   help="if content is mrc, start of the range of images to be processed")
    p.add_argument('--seq_end', type=int, default=None,
                   help="if content is mrc, end of the range of images to be processed")
    p.add_argument('--style_layers', '-sl', type=int, nargs='+', default=None,
                   help='Indices of VGG conv layers features of which are used to compute the loss on style.')
    p.add_argument('--style_layers_weights', '-slw', type=int, nargs='+', default=None,
                   help='Weight of each of the style-layers in the loss computation.')
    p.add_argument('--content_layers', '-cl', type=int, nargs='+', default=None,
                   help='Indices of VGG conv layers features of which are used to compute the loss on content.')
    p.add_argument('--content_layers_weights', '-clw', type=int, nargs='+', default=None,
                   help='Weight of each of the content-layers in the loss computation.')
    p.add_argument('--model_weights', '-mw', type=str, default='pretrained',
                   help='Weights of the NST model, one of {"pretrained", "random"}.')
    p.add_argument('--ncpus', '-n', type=int, default=0,
                   help='Number of CPU threads to use when device used is cpu. Ignored otherwise.')

    args = p.parse_args()
    start_time = time.time()
    
    # Reproducibility - added by FakET
    # run this script with PYTHONHASHSEED=0 python ...
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Read the following discussion to learn about benchmark and deterministic options
    # https://discuss.pytorch.org/t/what-is-the-differenc-between-cudnn-deterministic-and-cudnn-benchmark/38054
    # torch.backends.cudnn.deterministic=True - will only allow those CuDNN algorithms that are (believed to be) deterministic
    # torch.backends.cudnn.benchmark=True - the CuDNN library will benchmark several algorithms and pick that which it found to be fastest
    # so benchmark=True potentially leads to faster execution but larger memory peaks
    # warning: benchmarking may pick a different algorithm even with the deterministic flag is set
    # CONCLUSION: in order to be able to reproduce the results of our paper, for now we keep
    # benchmark=False and deterministic=True. In the future, this could/should change to optimize for speed
    # maybe could even become an argument? Let's see.
    torch.manual_seed(args.random_seed)
       
    # Implementing support for mrc file input
    # We normalize the data to interval [0, 1] as is expected by 
    # the NST. We do it on the whole mrc file to keep the relationships 
    # between each image. But we do not loose the precision because 
    # we stay in float32
    if args.content.endswith('.mrc'):
        input_type = 'mrc'
        
        assert args.output.endswith('.mrc'), \
        'If content is mrc file, output must also be mrc file.' \
        'Save every will be set to `IMG{:02d}_{}x{}_i{:05d}_mrcfname.png`automatically'
        
        content_mrc = normalize(load_mrc(args.content))
        assert len(content_mrc.shape) == 3, \
        'Content.mrc must be a sequence of 2D arrays.'
        
        # We only support one mrc style file, not a sequence
        assert args.styles[0].endswith('.mrc'), \
        'Style must be mrc file if content is mrc.'
        style_mrc_orig = load_mrc(args.styles[0])  # keeping not normalized
        style_mrc = normalize(style_mrc_orig)
        
        assert style_mrc.shape[0] == content_mrc.shape[0], \
        'First dimension of content and style must match.'
        
        if args.init.endswith('.mrc'):
            init_mrc = normalize(load_mrc(args.init))
            assert init_mrc.shape == content_mrc.shape, \
            'Shape of init must be the same as the shape of content.'
        else:
            init_mrc = None
        
    else:
        input_type = 'image'
        content_img = load_image(args.content, args.proof)
        style_imgs = [load_image(img, args.proof) for img in args.styles]
        assert not args.init.endswith('.mrc'), \
        'Init can be an mrc file only if content is also an mrc file.'
        init = args.init

    # Specifying callback and output image type 
    output_image_path = Path(args.output)
    if input_type == 'mrc':
        # Final image output type and path
        output_image_type = 'mrc'
        # Image type and output fname for callback
        # image_type = 'pil'  # standard greyscale png
        image_type = 'pil_cmap'  # greyscale image (mean of RGB) + viridis cmap
        p = Path(args.output)
        args.output = args.output.replace(p.name, 'IMG{:02d}_{}x{}_i{:05d}_' + p.stem + '.png')
    else:
        image_type = 'pil'
        if Path(args.output).suffix.lower() in {'.tif', '.tiff'}:
            image_type = 'np_uint16'
        output_image_type = image_type
    

    # Handling devices
    devices = [torch.device(device) for device in args.devices]
    if not devices:
        devices = [torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')]
    if len(set(device.type for device in devices)) != 1:
        print('Devices must all be the same type.')
        sys.exit(1)
    if not 1 <= len(devices) <= 2:
        print('Only 1 or 2 devices are supported.')
        sys.exit(1)
    print('Using devices:', ' '.join(str(device) for device in devices))

    if devices[0].type == 'cpu' and args.ncpus:
        torch.set_num_threads(args.ncpus) 
        print('CPU threads used:', torch.get_num_threads())
    if devices[0].type == 'cuda':
        for i, device in enumerate(devices):
            props = torch.cuda.get_device_properties(device)
            print(f'GPU {i} type: {props.name} (compute {props.major}.{props.minor})')
            print(f'GPU {i} VRAM:', round(props.total_memory / 1024 / 1024), 'MiB')

    # Handling end scale
    end_scale = int(args.end_scale.rstrip('+'))
    if args.end_scale.endswith('+'):
        end_scale = get_safe_scale(*content_img.size, end_scale)
    args.end_scale = end_scale

    # Web interface was removed to make this project more light-weight
    web_interface = None
    if args.web:
        raise NotImplementedError('WebInterface was removed from this fork.')

    for device in devices:
        torch.tensor(0).to(device)
        
        
    torch.manual_seed(args.random_seed)
    
    
    print('Loading model...')   
    st = StyleTransfer(devices=devices, 
                       pooling=args.pooling, 
                       style_layers=args.style_layers, 
                       content_layers=args.content_layers, 
                       style_layers_weights=args.style_layers_weights, 
                       content_layers_weights=args.content_layers_weights,
                       model_weights=args.model_weights) 
    
    output_image = []
    seq_len = 1 if input_type == 'image' else content_mrc.shape[0]
    
    seq_end = args.seq_end or seq_len
    assert seq_end <= seq_len, f'seq_end max is {seq_len}'
    for i in range(args.seq_start, seq_end):
        # print(f'Processing image {i}/{seq_len}')
        # Since the NST works on RGB images only, we have to 
        # copy the data 3x along the last axis
        get3ch = lambda x: np.broadcast_to(x[..., np.newaxis], x.shape + (3,))
        if input_type == 'mrc':
            content_img = get3ch(content_mrc[i])
            style_imgs = [get3ch(style_mrc[i])]
            init = args.init if init_mrc is None else get3ch(init_mrc[i])
            # Now the images are normalized and in RGB format like PIL
            # but in float dtype (so not loosing precision by going to int)
        
        # Running callback only for the first image in the sequence
        # callback = None
        # if i == 0:  
        callback = Callback(st, args, seq_i=i, seq_len=seq_len, image_type=image_type, web_interface=web_interface)
        atexit.register(callback.close)

        defaults = StyleTransfer.stylize.__kwdefaults__
        st_kwargs = {k: v for k, v in args.__dict__.items() if k in defaults and k != 'init'}
        try:
            st.stylize(content_img, style_imgs, init=init, **st_kwargs, callback=callback)
        except KeyboardInterrupt:
            break

        output_image.append(st.get_image(output_image_type))
        
        # Save the config in the same folder as the output
        os.makedirs(output_image_path.parent, exist_ok=True)
        with open(output_image_path.with_suffix('.json'), 'w') as fp:
            json.dump(callback.get_trace(), fp, indent=4)

    if input_type == 'mrc':
        # Saving the full sequence of mrc outputs
        output_image = match_mean_std(np.array(output_image), style_mrc_orig[args.seq_start:seq_end])
        save_mrc(output_image, output_image_path, overwrite=True)
    else:
        # Saving one image output
        if output_image[0] is not None:
            save_image(output_image_path, output_image[0])
    print(f'Duration: {(time.time() - start_time) / 60:.2f} minutes.')
    


if __name__ == '__main__':
    try:
        # Just identification of the script when running in parallel on multiple GPUs
        print('PID: {}, EXPNAME: {}, CUDA_VISIBLE_DEVICES: {}'.format(
        os.getpid(), os.environ['EXPNAME'], os.environ['CUDA_VISIBLE_DEVICES']))
    except:
        pass
    main()
