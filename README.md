# SpikCommander

## Dependencies

### Python

Please use Python 3.9 and above ```python>=3.9```

### SpikingJelly

Install SpikingJelly using:

```
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
python setup.py install --user
```

Installing SpikingJelly using ```pip``` is not yet compatible with this repo.

### Other

Install the other dependencies from the requirements.txt file using:

```
pip install -r requirements.txt
```


## Usage

The first thing to do after installing all the dependencies is to specify the `datasets_path` in `config.py`. Simply create an empty data directory, preferably with three subdirectories for SHD, SSC, and GSC. The `datasets_path` should correspond to these subdirectories. The datasets will then be downloaded and preprocessed automatically. 

For example:

```
cd SpikCommander
mkdir -p Datasets/SHD
mkdir -p Datasets/SSC
mkdir -p Datasets/GSC
```

To train a new model as defined by the ```config.py``` simply use:

```
python main_former_shd_spikcommander.py
python main_former_ssc_spikcommander.py
python main_former_gsc_spikcommander.py
```

The loss and accuracy for the training and validation at every epoch will be printed to ```stdout``` and the best model will be saved to the current directory.
If the ```use_wandb``` parameter is set to ```True```, a more detailed log will be available at the wandb project specified in the configuration.

### Our organized code will be made publicly available in a common repository once the camera-ready version of the submission paper is finalized.