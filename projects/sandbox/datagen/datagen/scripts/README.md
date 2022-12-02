# BBHNet generate-waveforms
This project generates the specified number of raw waveforms; i.e., waveforms that have not been projected onto an interferometer. The waveforms are saved to an HDF5 file, along with the parameters that define them, which were sampled in the process.

## Quickstart Instructions
These instructions assume that `poetry` and `pinto` have been installed as per the instructions here: [https://github.com/ML4GW/pinto#readme](https://github.com/ML4GW/pinto#readme). To install the environment necessary to run this project, run 
```
$ pinto build
```
from the `datagen` directory.

Once installed, running 
```
$ pinto run generate-waveforms --typeo ..:generate-waveforms
```
from your base `conda` environment and in the `datagen` directory will run the script with the default arguments defined in the `pyproject.toml` in the `sandbox` directory. Here, [`typeo`](https://github.com/ML4GW/typeo) has been used to pass the arguments specified in the `generate-waveforms` section of `../pyproject.toml` to the `generate-waveforms` command. 

## `generate-waveforms` Configuration
The following is the `help` message for `generate-waveforms`:
```
usage: main [-h] --prior PRIOR --n-samples N_SAMPLES --reference-frequency
            REFERENCE_FREQUENCY --minimum-frequency MINIMUM_FREQUENCY
            --sample-rate SAMPLE_RATE --waveform-duration WAVEFORM_DURATION
            --waveform-approximant WAVEFORM_APPROXIMANT --logdir LOGDIR
            --datadir DATADIR [--force-generation] [--verbose]

Simulates a set of raw BBH signals and saves them to an output file.

optional arguments:
  -h, --help            show this help message and exit
  --prior PRIOR         function returning a bilby PriorDict for bilby to
                        sample from (default: None)
  --n-samples N_SAMPLES
                        number of signals to simulate (default: None)
  --reference-frequency REFERENCE_FREQUENCY
                        reference frequency for waveform generation (default:
                        None)
  --minimum-frequency MINIMUM_FREQUENCY
                        minimum frequency for waveform generation (default:
                        None)
  --sample-rate SAMPLE_RATE
                        sample rate of the simulated signals (default: None)
  --waveform-duration WAVEFORM_DURATION
                        length of injected waveforms in seconds (default:
                        None)
  --waveform-approximant WAVEFORM_APPROXIMANT
                        which lalsimulation waveform approximant to use
                        (default: None)
  --logdir LOGDIR       directory where log file will be written (default:
                        None)
  --datadir DATADIR     output directory to which signals will be written
                        (default: None)
  --force-generation    if True, generate signals even if path already exists
                        (default: False)
  --verbose             log verbosely (default: False)
```
