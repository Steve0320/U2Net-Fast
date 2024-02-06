import os
import glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import Normalize, ToPILImage
import torch.nn.functional as F
from torch.multiprocessing import Pool
import itertools
from PIL import Image

from .model import U2NET
from .dataloader import U2Dataset


# Helper - write out the given tensor data to an image
def write_image(filename, data, out_dir, extension):
    image_name = os.path.splitext(os.path.basename(filename))[0]
    path = os.path.join(out_dir, image_name) + os.extsep + extension
    ToPILImage()(data).save(path)


# The main utility class - wraps up loading, inference, and writing
class Remover:

    def __init__(self,
                 model_path=os.path.join(os.getcwd(), 'models', 'u2net.pth'),
                 write_concurrency=torch.multiprocessing.cpu_count(),
                 dataloader_workers=4,
                 batch_size=5,
                 pin_memory=False,
                 background_fill=None,
                 output_format=None,
                 threshold=0.5,
                 apply_mask=True,
                 save_output=True):

        self.model_path = model_path
        self.write_concurrency = write_concurrency
        self.dataloader_workers = dataloader_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.threshold = threshold
        self.apply_mask = apply_mask
        self.write_output = save_output

        self.background_fill = [0, 0, 0] if background_fill is None else background_fill

        # Auto-enable transparency support if alpha channel provided
        self.enable_transparency = (len(self.background_fill) == 4)

        if output_format is None:
            output_format = 'png' if self.enable_transparency else 'jpg'
        self.output_format = output_format

    # Process a batch yielded by the dataloader. This overwrites sample['image'] with the calculated mask.
    def process_batch(self, sample, model, image_size):

        # Process loaded data. Dimensions BxCxHxW
        o_images = sample['image']

        # Move full image to GPU for processing. We don't expand to float until we get to the GPU to
        # save on some host memory. We also keep a copy of the original image in host memory so that we
        # don't have to reload it later.
        images = o_images.to('cuda').float()
        del sample['image']

        # Downsample to the expected 320x320 model input. This also discards the original image reference
        # since it's very large and we no longer need it.
        images = F.interpolate(images, size=(320, 320), mode='bilinear')

        # Squash values to 0-1 and apply standard data normalization using ImageNet values
        images = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(images / images.max())

        with torch.no_grad():
            d1, d2, d3, d4, d5, d6, d7 = model(images)

        del images, d2, d3, d4, d5, d6, d7

        # Normalize (again) from 0-1, expand to full image size, and shuttle back to host memory
        d_min, d_max = d1.aminmax()

        sample['mask'] = F.interpolate((d1 - d_min) / (d_max - d_min), size=image_size, mode='bilinear')
        del d1

        sample['mask'] = (sample['mask'] * 255).byte().to('cpu')
        sample['image'] = o_images

    # Apply the mask to the sample. This step takes place on the CPU since it's a pretty quick operation, and the time
    # to move the full image back into the GPU generally is longer anyway.
    def mask_batch(self, sample, image_size):

        # Expand the given background to the dimensions of the output (permuted to CxHxW format)
        rgb = torch.tensor(self.background_fill, dtype=torch.uint8).expand(image_size[0], image_size[1], len(self.background_fill)).permute(2, 0, 1)

        # Add transparency dimension to original image if needed
        if self.enable_transparency:
            sample['image'] = F.pad(sample['image'], (0, 0, 0, 0, 0, 1), 'constant', 255)

        sample['image'] = torch.where(sample['mask'] > int(self.threshold * 255), sample['image'], rgb)
        del rgb

    # Helper - image writing with PIL can be slow, so parallelize the operations.
    def write_batch(self, sample, output_dir):
        with Pool(self.write_concurrency) as p:
            p.starmap(write_image, zip(sample['filename'], sample['image'], itertools.repeat(output_dir), itertools.repeat(self.output_format)))

    # Main entry point - take in a folder of images, run inference, optionally apply mask, and write output.
    def batch_remove_background(self,
                                input_dir=os.path.join(os.getcwd(), 'inputs'),
                                output_dir=os.path.join(os.getcwd(), 'outputs'),
                                image_size=None,
                                show_progress=True):

        image_names = glob.glob(input_dir + os.sep + '*')
        print(f"Running inference on {len(image_names)} images...")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if image_size is None:
            img = Image.open(image_names[0])
            w, h = img.size
            image_size = (h, w)

        # Load model
        print("Loading model...")
        net = U2NET(3, 1)
        net.load_state_dict(torch.load(self.model_path))
        net.cuda().eval()

        # Prepare data loader
        dataset = U2Dataset(image_name_list=image_names)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.dataloader_workers, pin_memory=self.pin_memory)

        def p(bs):
            self.process_batch(bs, net, image_size)
            if self.apply_mask:
                self.mask_batch(bs, image_size)
            else:
                bs['image'] = bs['mask']

            if self.write_output:
                self.write_batch(bs, output_dir)

            del bs

        if show_progress:
            with tqdm(total=len(dataset)) as pbar:
                for batch_sample in dataloader:
                    p(batch_sample)
                    pbar.update(dataloader.batch_size)

        else:
            for batch_sample in dataloader:
                p(batch_sample)
