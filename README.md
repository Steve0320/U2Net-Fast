# U2Net-Fast

This library builds on the [U2Net machine learning model](https://github.com/xuebinqin/U-2-Net) to provide very fast
background removal for large images. This is primarily accomplished by rewriting the data preparation steps to more
effectively utilize the GPU. Support for multiple DataLoader workers and batching has also been added.

### IMPORTANT!
This library requires that all input images in a batch be the same size! This library does not automatically resize
or align inputs. If you need to process mixed resolution inputs, you should either (a) set the batch size to 1 to ensure
all images are in their own batch at the cost of some performance, or (b) pad out the dimensions of any undersized images
before running them through this library.

Also note that this library requires a fairly large amount of VRAM for large images/batch sizes. The total amount of VRAM
needed can be calculated as follows: (3 * batch_size * height * width * 4). This is because each image must be expanded into
a 3-channel (RGB) tensor of floats (4 bytes) in order to perform the necessary rescaling. The original U2Net repository does
this rescaling on the CPU, so it uses much less memory (but is significantly slower).

## Installation
This library is available on the Python Package Index (PyPI) under the name "u2net_fast".

## Prerequisites
The following libraries are required:
- torch
- torchvision
- tqdm
- pillow (or pillow-simd, which can give a significant performance boost for certain output types)

Installing via the prebuilt package should automatically install all prerequisites.

## Usage
For example scripts, see the `examples` directory. Note that for all example scripts, the U2Net model weights must be
downloaded to the `examples/models` directory. They may be obtained from the original U2Net repository following the instructions
in the [Usage for Salient Object Detection](https://github.com/xuebinqin/U-2-Net?tab=readme-ov-file#usage-for-salient-object-detection)
section.

The core component of this library is the Remover class, which may be imported using:
```
from u2net_fast.remover import Remover
```
This class exposes several methods, though for most use cases only the `batch_remove_background` function will be needed.

The Remover class constructor takes several arguments to customize behavior. They are listed below:
- model_path (defaults to {os.getcwd()}/models/u2net.pth)
  - The path pointing to the U2Net model weights.
- write_concurrency (defaults to the number of CPU cores)
  - The number of parallel workers used to write out the output images. One of the slowest steps in this library is output image encoding, so paralleling this operation as much as possible usually gives significant speedups.
- dataloader_workers (defaults to 4)
  - The number of PyTorch workers to assign to loading images.
- batch_size (defaults to 5)
  - The number of images to load/process each batch. If you are running out of VRAM, try reducing this size.
- pin_memory (defaults to False)
  - Whether the PyTorch dataloader should pin memory. In tests this actually seemed to slightly slow down performance for whatever reason. Other systems may have better luck.
- background_fill (defaults to [0, 0, 0])
  - The RGB or RGBA color used to fill the background if apply_mask=True. If this is set to a 4 element array, the output_format must support transparency. 
- output_format (defaults to 'jpg' if background_fill is 3 elements, or 'png' if background_fill is 4 elements)
  - The format used to write output images. This is also used as the extension for the output files. This should be a format that PIL recognizes. 'jpg' is usually significantly faster than 'png'.
- threshold (defaults to 0.5)
  - The U2Net mask output is grayscale. This value (between 0 and 1) determines the threshold used to determine whether something is "in" or "out" of the mask at the boundary. This rarely needs to be changed. 
- apply_mask (defaults to True)
  - Whether the calculated mask should be applied to the model. If this is False, the output from `batch_remove_background` will be the image masks instead of the actual masked image. If you only need masks this can lead to a significant speedup.
- save_output (defaults to True)
  - Whether the final output should be saved. This almost always should be true. The only time it's useful to set to False is when benchmarking performance.

Once the remover object is instantiated, you may call `batch_remove_background` on it like so:
```
r = Remover()
r.batch_remove_background()
```

The `batch_remove_background` function also accepts several arguments, detailed below:
- input_dir (defaults to {os.getcwd()/inputs)
  - The folder to load inputs in from. This should be a folder full of images with the same dimensions
- output_dir (defaults to {os.getcwd()/outputs)
  - The folder to save outputs to. This is not used if save_output=False. It will be automatically created if it does not exist.
- image_size (defaults to the size of the first image in the input_dir)
  - The size of the images. This is automatically inferred from the first input image but can be set manually if desired.
- show_progress (defaults to True)
  - Whether a TQDM progress bar should be shown during inference. Set this to False to disable progress tracking.

## Advanced Usage
While the `batch_remove_background` function is the easiest way to use this library, the different stages of the processing
pipeline can also be called manually if more fine-grained control is needed. There are three main functions that `batch_remove_background`
calls, listed below. These functions can all be called on an instance of a Remover object.

#### process_batch
This function takes in a batch sample, a U2Net model, and a tuple representing the image size. It modifies the input sample
in place and adds the calculated masks under the 'mask' key.

#### apply_mask
This function takes in a batch sample and a tuple representing the image size and applies the mask to the image. The 'image'
key in the sample is overwritten with the new masked image.

#### write_batch
This function takes in a batch sample and an output directory and writes the image data in the 'image' key to disk.

To use these functions effectively, you should initialize a dataset and dataloader like so:
```
from u2net_fast.model import U2NET
from u2net_fast.dataloader import U2Dataset

dataset = U2Dataset(image_name_list=image_names)
dataloader = DataLoader(dataset, batch_size=1 shuffle=False)
```

U2Dataset and DataLoader are just subclasses of PyTorch's Dataset and Dataloader, so all the standard arguments are accepted.
The DataLoader's batch_size should be set to 1 or greater - the pipeline functions expect the batch_sample to contain a batch
dimension. Then, set up a loop to yield samples:
```
for batch_sample in dataloader:
  <call the pipeline functions as needed here>
```

## Performance
Performance was measured on a relatively modest system (an i7 7700k, and a GTX1080). At the default settings, processing
a batch of 125 images with mask application enabled (apply_mask=True) took approximately 33 seconds, with a peak throughput of about 4.5
images/second.

Without mask application enabled (apply_mask=False), the same batch took approximately 13 seconds with a peak throughput of
about 9 images/second. This performance difference was mostly due to the additional data that needed to be saved to disk when
outputting the full masked images (3 channels for the full RGB images vs 1 channel for the grayscale masks), _not_ the actual
mask application step.

With result saving disabled (save_output=False), and mask application disabled, the same batch took approximately 10 seconds
with a peak throughput of 11.45 images/second. With mask application enabled, the same batch took approximately 14 seconds
with a peak throughput of about 11 images/second (though this peak was reached much later).