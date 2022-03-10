import glob
import h5py
import os

import numpy as np

from gwpy.timeseries import TimeSeries
from gwpy.segments import Segment, SegmentList

import utils as utils


'''
Tools to generate a dataset of glitches from omicron triggers.

For information on how the omicron triggers were generated see: /home/ethan.marx/bbhnet/generate-glitch-dataset/omicron/12566/H1L1_1256665618_100000/runfiles/omicron_params_H1.txt on CIT cluster
for an example omicron parameter file. 

Of note is the clustering timescale of 1 second

'''




def generate_glitch_dataset(
    ifo: str,
    snr_thresh: float,
    start: float,
    stop: float,
    window: float,
    sample_rate: float = 4096,
    omicron_dir: str ='/home/ethan.marx/bbhnet/generate-glitch-dataset/omicron/',
    vetoes: SegmentList = None,
):

    """
    Generates a list of omicron trigger times that satisfy snr threshold
    
    Arguments:
    - ifo: ifo to generate glitch triggers for
    - snr_thresh: snr threshold above which to keep as glitch
    - start: start gpstime
    - stop: stop gpstime
    - window: half window around trigger time to query data for
    - sample_rate: sampling frequency
    - omicron_dir: base directory of omicron triggers (see /home/ethan.marx/bbhnet/generate-glitch-dataset/omicron/)
    """
     
    glitches = [] 
    snrs = []
    
    # omicron triggers are split up by directories into segments of 10^5 seconds
    # get paths for relevant directories based on start and stop gpstimes
    
    gps_day_start = str(start)[:5]
    gps_day_end = str(stop)[:5] 
    all_gps_days = np.arange(int(gps_day_start), int(gps_day_end) + 1, 1) 

    # loop over gps days
    for i, day in enumerate(all_gps_days):
        
        # get path for this gps day 
        omicron_path = omicron_dir + f'{day}/'
        
        # do we want pre or post vetoes? I presume we want to use some glitches found in vetoes (CAT3, not CAT1)
        trig_file = glob.glob(omicron_path + f'/*/PostProc/unclustered/triggers_unclustered_{ifo}.txt')[0]
        
                          
        # load in triggers 
        triggers = np.loadtxt(trig_file)
        print(len(triggers)) 
        # if passed, apply vetos
        if vetoes is not None:
            not_vetoed_args = utils.veto(triggers[:,0], vetoes)
            triggers = triggers[not_vetoed_args]
        print(len(triggers))
        # restrict triggers to within gps start and stop times
        times = triggers[:,0]
        time_args = np.logical_and(times > start, times < stop)
        triggers = triggers[time_args]
        
        # apply snr thresh
        day_snrs = triggers[:,2] # second column is snrs
        snr_thresh_args = np.where(day_snrs > snr_thresh) 
        triggers = triggers[snr_thresh_args] 
    
        snrs.extend(day_snrs) 
        
        # query data for each trigger
        for trigger in triggers:
            time = trigger[0]
            trig_data = TimeSeries.fetch_open_data(ifo, time-window, time+window) 
            
            glitches.append(trig_data) 
        
    return glitches, snrs
 
def main(
    snr_thresh: float,
    start: float,
    stop: float,
    window: float,
    sample_rate: float = 4096,
    outdir: str = './',
    omicron_dir: str ='/home/ethan.marx/bbhnet/generate-glitch-dataset/omicron/',
    H1_veto_file: str=None,
    L1_veto_file: str=None

):
   
    """Simulates a set of glitches for both H1 and L1 that can be added to background
    Arguments:
    
    - snr_thresh: snr threshold above which to keep as glitch
    - start: start gpstime
    - stop: stop gpstime
    - window: half window around trigger time to query data for
    - sample_rate: sampling frequency
    - outdir: output directory to which signals will be written
    - omicron_dir: base directory of omicron triggers (see /home/ethan.marx/bbhnet/generate-glitch-dataset/omicron/)
    - H1_veto_file: path to file containing vetoes for H1
    - L1_veto_file: path to file containing vetoes for L1
    """ 
    
    # if passed, load in H1 vetoes and convert to gwpy SegmentList object 
    if H1_veto_file is not None:
        
        # load in H1 vetoes
        H1_vetoes = np.loadtxt(H1_veto_file)
        
        # convert arrays to gwpy Segment objects
        H1_vetoes = [Segment(seg[0], seg[1]) for seg in H1_vetoes]
        
        # create SegmentList object
        H1_vetoes = SegmentList(H1_vetoes).coalesce()
    
    if L1_veto_file is not None:
    
        L1_vetoes = np.loadtxt(L1_veto_file)
        L1_vetoes = [Segment(seg[0], seg[1]) for seg in L1_vetoes]
        L1_vetoes = SegmentList(L1_vetoes).coalesce()
    

    H1_glitches, H1_snrs = generate_glitch_dataset(
        'H1',
        snr_thresh,
        start,
        stop,
        window,
        sample_rate,
        vetoes=H1_vetoes
    )
   
    L1_glitches, L1_snrs = generate_glitch_dataset(
        'L1',
        snr_thresh,
        start,
        stop,
        window,
        sample_rate,
        vetoes=L1_vetoes
    )
    
    glitch_file = os.path.join(outdir, 'glitches.h5') 
    
    with h5py.File(glitch_file, 'w') as f:
        # not sure what format we want here
        f.create_dataset("H1_glitches", data=H1_glitches)
        f.create_dataset("H1_snrs", data = H1_snrs)
        
        f.create_dataset("L1_glitches", data=L1_glitches)
        f.create_dataset("L1_snrs", data = L1_snrs)
       
    return glitch_file

if __name__ == "__main__":
    H1_veto_file = '/home/olib/public_html/Ethan/O3b_offline/back_train_lowfreq/vetoes/H1Cat1.txt'
    L1_veto_file = '/home/olib/public_html/Ethan/O3b_offline/back_train_lowfreq/vetoes/L1Cat1.txt'
    main(snr_thresh=10, start=1256665622 , stop=1256699999, window=1, sample_rate=4096, H1_veto_file=H1_veto_file, L1_veto_file=L1_veto_file) 
