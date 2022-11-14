# BBHNet generate-waveforms
This project uses BBHNet's `injection` library to generate the specified number of raw waveforms; i.e., waveforms that have not been projected onto an interferometer. The waveforms are saved to an HDF5 file, along with the parameters that define them, which were sampled in the process.

## Quickstart Instructions
These instructions assume that `poetry` and `pinto` have been installed as per the instructions here (NEED A LINK TO POETRY/PINTO INSTALLATION INSTRUCTIONS). To install the environment necessary to run this project, run 
```
$ pinto build
```
from the `generate_waveforms` directory.

Once installed, running 
```
$ pinto run generate-waveforms --typeo ..:generate-waveforms
```
from your base `conda` environment and in the `generate_waveforms` directory will run the script with the default arguments defined in the `pyproject.toml` in the `sandbox` directory.

For an explanation of the anatomy of the above command, see here (GUIDE EXPLAINING HOW PINTO WORKS)

## `generate-waveforms` Configuration
The following is the `help` message for `generate-waveforms`:
```
usage: main [-h] --prior PRIOR --n-samples N_SAMPLES --logdir LOGDIR --datadir
            DATADIR --reference-frequency REFERENCE_FREQUENCY
            --minimum-frequency MINIMUM_FREQUENCY --sample-rate SAMPLE_RATE
            --waveform-duration WAVEFORM_DURATION
            [--waveform-approximant WAVEFORM_APPROXIMANT] [--force-generation]
            [--verbose]

Simulates a set of raw BBH signals and saves them to an output file.

optional arguments:
  -h, --help            show this help message and exit
  --prior PRIOR         function returning a bilby PriorDict for bilby to
                        sample from (default: None)
  --n-samples N_SAMPLES
                        number of signals to simulate (default: None)
  --logdir LOGDIR       directory where log file will be written (default:
                        None)
  --datadir DATADIR     output directory to which signals will be written
                        (default: None)
  --reference-frequency REFERENCE_FREQUENCY
                        reference frequency for waveform generation (default:
                        None)
  --minimum-frequency MINIMUM_FREQUENCY
                        minimum_frequency for waveform generation (default:
                        None)
  --sample-rate SAMPLE_RATE
                        rate at which to sample waveforms (default: None)
  --waveform-duration WAVEFORM_DURATION
                        length of injected waveforms in seconds (default:
                        None)
  --waveform-approximant WAVEFORM_APPROXIMANT
                        which lalsimulation waveform approximant to use
                        (default: IMRPhenomPv2)
  --force-generation    if True, generate signals even if path already exists
                        (default: False)
  --verbose             log verbosely (default: False)
```
