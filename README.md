## Setup:

```bash
git clone git@github.com:cms-denoising/SimDenoising_training
cd SimDenoising_training
[launch conda/singularity environment]
HOME=$PWD pip install --user --no-cache-dir magiconfig mplhep
```

## Description of files:

dataset.py: loads image data from a root file into a dataset object to be used in training

models.py: the CNN and the loss function(s)

train.py: trains the network, accepts the following command line arguments
* --num_of_layers
* --outf
* --epochs
* --lr
* --trainfileSharp
* --trainfileFuzz
* --valfileSharp
* --valfileFuzz
* --batchSize
* --model
* --patchSize
* --kernelSize
* --features
* --transform
* --num-workers
* --randomseed

loss_plot.py: plots training and validation losses

output.py: creates npz file of output images from running trained network on low-quality input images

analysis_plots.py: plots sample images, histograms, and scatterplots for various physical quantities
