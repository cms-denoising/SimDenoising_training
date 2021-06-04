## Setup:

```bash
git clone git@github.com:kpedro88/SimDenoising_training
cd SimDenoising_training
[launch conda/singularity environment]
HOME=$PWD pip install --no-cache-dir magiconfig
```

## Description of files:

dataset.py - loads 100*100 pixel data from a root file into a Dataset object to be used in training

train.py - trains the network, accepts the following command line arguments
	 --num_of_layers
	 --sigma
	 --outf
	 --epochs
	 --lr
	 --trainfile
	 --valfile
	 --batchSize
	 --model
	 --patchSize
	 --kernelSize
	 --features

models.py - the CNN and the loss function(s)

in tools/: various unfinished and undocumented tools for making plots, etc. 
