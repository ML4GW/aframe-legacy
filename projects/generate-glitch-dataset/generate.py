import numpy as np
import glob
import hdf5
from gwpy.timeseries import TimeSeries

# location of omicron triggers; this is the base path for oLIB offline run (on CIT)
# we can copy omicron triggers to a BBH specific location if we want
omicron_base_path = '/home/olib/public_html/Ethan/O3b_offline/0lag_lowfreq/'

# the start and stop times for which we want glitches (in our case, O3 replay times)
start = 1262304000
stop = 1265760000

# threshold above which we declare glitch
snr_thresh = 8

# length of timeseries (i.e. window of network)
ts_window = 1 # seconds

# sampling frequency
sample_rate = 1024 # Hz 

# omicron triggers are split up into segments of 10^5 seconds
# get paths for each of these directories
omicron_day_paths = glob.glob(base_path + '/12*/')

# loop over gps days
for i, path in enumerate(omicron_day_paths):
    
    gps_day = path.split('/')[-2]
    
    # do we want pre or post vetoes? I presume we want to use some glitches found in vetoes
    omicron_trigger_files = glob.glob(dir_ + '/*/PostProc/clustered/*.txt') 
    
    for trig_file in omicron_trigger_files:
                      
        # load in triggers 
        triggers = np.loadtxt(trig_file)
        
        # apply snr thresh
        snrs = triggers[:,2] # second column is snrs
        snr_thresh_args = np.where(snrs > snr_thresh) 
        triggers = triggers[snr_thresh_args] 
   
              
        
        
    
    with h5py.File(f'{gps_day}_glitches.hdf5', 'w') as f:
     
