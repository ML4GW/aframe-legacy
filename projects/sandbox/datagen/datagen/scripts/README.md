# BBHNet datagen
This project generates all the data that BBHNet needs to run. It has four components:
1. generate-background: Pulls contiguous segments of background data from Hanford and Livingston between the given start and stop GPS times
2. generate-waveforms: Creates the specified number of raw waveforms; i.e., waveforms that have not been projected onto an interferometer. The parameters used to create these waveforms are also saved
3. generate-glitches: Uses `pyomicron` to identify glitches in each interferometer over the specified time period and saves segments of time around each glitch
4. generate-timeslides: Makes timeslides of the testing background data and injects waveforms on top of them

## Quickstart Instructions
These instructions assume that `poetry` and `pinto` have been installed as per the instructions here: [https://github.com/ML4GW/pinto#readme](https://github.com/ML4GW/pinto#readme). To install the environment necessary to run this project, run 
```
$ pinto build
```
from the `datagen` directory.

Once installed, running 
```
$ pinto run generate-background --typeo ..:generate-background
```
from your base `conda` environment and in the `datagen` directory will run the script with the default arguments defined in the `pyproject.toml` in the `sandbox` directory. Here, [`typeo`](https://github.com/ML4GW/typeo) has been used to pass the arguments specified in the `generate-background` section of `../pyproject.toml` to the `generate-background` command. The same format can be used to run any of `generate-waveforms`, `generate-glitches`, or `generate-timeslides`.

## `generate-background` Configuration
The following is the `help` message for `generate-background`:
```
usage: main [-h] --start START --stop STOP --ifos IFOS [IFOS ...]
            --sample-rate SAMPLE_RATE --channel CHANNEL --frame-type
            FRAME_TYPE --state-flag STATE_FLAG --minimum-length MINIMUM_LENGTH
            --logdir LOGDIR --datadir DATADIR [--force-generation] [--verbose]

Generates background data for training BBHnet

optional arguments:
  -h, --help            show this help message and exit
  --start START         starting GPS time of the time period to analyze
                        (default: None)
  --stop STOP           ending GPS time of the time period to analyze
                        (default: None)
  --ifos IFOS [IFOS ...]
                        which ifos to get background data for (default: None)
  --sample-rate SAMPLE_RATE
                        sample rate of the queried data (default: None)
  --channel CHANNEL     strain channel to query data from (default: None)
  --frame-type FRAME_TYPE
                        frame type for data discovery (default: None)
  --state-flag STATE_FLAG
                        name of segments to query from segment database
                        (default: None)
  --minimum-length MINIMUM_LENGTH
                        minimum length of contiguous segment to save (default:
                        None)
  --logdir LOGDIR       directory where log file will be written (default:
                        None)
  --datadir DATADIR     output directory to which background data will be
                        written (default: None)
  --force-generation    if True, query data even if a path already exists
                        (default: False)
  --verbose             log verbosely (default: False)
```

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

## `generate-glitches` Configuration
The following is the `help` message for `generate-glitches`:
```
usage: main [-h] --snr-thresh SNR_THRESH --start START --stop STOP --test-stop
            TEST_STOP --q-min Q_MIN --q-max Q_MAX --f-min F_MIN --cluster-dt
            CLUSTER_DT --chunk-duration CHUNK_DURATION --segment-duration
            SEGMENT_DURATION --overlap OVERLAP --mismatch-max MISMATCH_MAX
            --window WINDOW --logdir LOGDIR --datadir DATADIR --channel
            CHANNEL --frame-type FRAME_TYPE --sample-rate SAMPLE_RATE
            --state-flag STATE_FLAG --ifos IFOS [IFOS ...]
            [--veto-files VETO_FILES [VETO_FILES ...]] [--force-generation]
            [--verbose]

Generates a set of glitches for both
        H1 and L1 that can be added to background

        First, an omicron job is launched via pyomicron
        (https://github.com/gwpy/pyomicron/). Next, triggers (i.e. glitches)
        above a given SNR threshold are selected, and data is queried
        for these triggers and saved in an h5 file.

optional arguments:
  -h, --help            show this help message and exit
  --snr-thresh SNR_THRESH
                        snr threshold above which to keep as glitch (default:
                        None)
  --start START         start gpstime (default: None)
  --stop STOP           training stop gpstime (default: None)
  --test-stop TEST_STOP
                        testing stop gpstime (default: None)
  --q-min Q_MIN         minimum q value of tiles for omicron (default: None)
  --q-max Q_MAX         maximum q value of tiles for omicron (default: None)
  --f-min F_MIN         lowest frequency for omicron to consider (default:
                        None)
  --cluster-dt CLUSTER_DT
                        time window for omicron to cluster neighboring
                        triggers (default: None)
  --chunk-duration CHUNK_DURATION
                        duration of data (seconds) for PSD estimation
                        (default: None)
  --segment-duration SEGMENT_DURATION
                        duration of data (seconds) for FFT (default: None)
  --overlap OVERLAP     overlap (seconds) between neighbouring segments and
                        chunks (default: None)
  --mismatch-max MISMATCH_MAX
                        maximum distance between (Q, f) tiles (default: None)
  --window WINDOW       half window around trigger time to query data for
                        (default: None)
  --logdir LOGDIR       directory where log file will be written (default:
                        None)
  --datadir DATADIR     output directory to which signals will be written
                        (default: None)
  --channel CHANNEL     channel name used to read data (default: None)
  --frame-type FRAME_TYPE
                        frame type for data discovery w/ gwdatafind (default:
                        None)
  --sample-rate SAMPLE_RATE
                        sampling frequency of timeseries data (default: None)
  --state-flag STATE_FLAG
                        identifier for which segments to use (default: None)
  --ifos IFOS [IFOS ...]
                        which ifos to generate glitches for (default: None)
  --veto-files VETO_FILES [VETO_FILES ...]
                        dictionary where key is ifo and value is path to file
                        containing vetoes (default: None)
  --force-generation    if True, query data even if a path already exists
                        (default: False)
  --verbose             log verbosely (default: False)
```

## `generate-timeslides` Configuration
```
The following is the `help` message for `generate-timeslides`:
usage: main [-h] --start START --stop STOP --logdir LOGDIR --datadir DATADIR
            --prior PRIOR --spacing SPACING --jitter JITTER --buffer- BUFFER_
            --n-slides N_SLIDES --shifts SHIFTS [SHIFTS ...] --ifos IFOS
            [IFOS ...] --minimum-frequency MINIMUM_FREQUENCY --highpass
            HIGHPASS --sample-rate SAMPLE_RATE --frame-type FRAME_TYPE
            --channel CHANNEL [--min-segment-length MIN_SEGMENT_LENGTH]
            [--chunk-length CHUNK_LENGTH]
            [--waveform-duration WAVEFORM_DURATION]
            [--reference-frequency REFERENCE_FREQUENCY]
            [--waveform-approximant WAVEFORM_APPROXIMANT]
            [--fftlength FFTLENGTH] [--state-flag STATE_FLAG]
            [--force-generation] [--verbose]

Generates timeslides of background and background + injections.
    Timeslides are generated on a per segment basis: First, science segments
    are queried for each ifo and coincidence is performed.
    To create a timeslide, each continuous segment is circularly shifted.

optional arguments:
  -h, --help            show this help message and exit
  --start START         starting GPS time of time period to analyze (default:
                        None)
  --stop STOP           ending GPS time of time period to analyze (default:
                        None)
  --logdir LOGDIR       directory where log file will be written (default:
                        None)
  --datadir DATADIR     output directory to which background data will be
                        written (default: None)
  --prior PRIOR         a prior function defined in prior.py script in the
                        injection lib (default: None)
  --spacing SPACING     spacing between consecutive injections (default: None)
  --jitter JITTER       level of random noise to add to the injection times
                        (default: None)
  --buffer- BUFFER_
  --n-slides N_SLIDES   number of timeslides (default: None)
  --shifts SHIFTS [SHIFTS ...]
                        List of shift multiples for each ifo. Will create
                        n_slides worth of shifts, at multiples of shift. If 0
                        is passed, will not shift this ifo for any slide.
                        (default: None)
  --ifos IFOS [IFOS ...]
                        List interferometers (default: None)
  --minimum-frequency MINIMUM_FREQUENCY
                        minimum_frequency used for waveform generation
                        (default: None)
  --highpass HIGHPASS   frequency at which data is highpassed (default: None)
  --sample-rate SAMPLE_RATE
                        sample rate of background data and waveforms (default:
                        None)
  --frame-type FRAME_TYPE
                        frame type for data discovery (default: None)
  --channel CHANNEL     strain channel to analyze (default: None)
  --min-segment-length MIN_SEGMENT_LENGTH
                        segments shorter than this time will be skipped
                        (default: None)
  --chunk-length CHUNK_LENGTH
                        if a segment is longer than chunk_length, it will be
                        split into chunks that are at most chunk_length long
                        (default: None)
  --waveform-duration WAVEFORM_DURATION
                        length of injected waveforms (default: 8)
  --reference-frequency REFERENCE_FREQUENCY
                        reference frequency for generating waveforms (default:
                        20)
  --waveform-approximant WAVEFORM_APPROXIMANT
                        waveform model to inject (default: IMRPhenomPv2)
  --fftlength FFTLENGTH
                        fftlength for calculating psd (default: 2)
  --state-flag STATE_FLAG
                        name of segments to query from segment database
                        (default: None)
  --force-generation    if True, query data even if a path already exists
                        (default: False)
  --verbose             log verbosely (default: False)
```
