import numpy as np


# function modified from hveto: https://github.com/gwdetchar/hveto/blob/7974abee50560b86f4d22ec2d6096117ed1a132d/hveto/core.py
# needs some unit tests!
def veto(times, segmentlist):
    
    """Remove events from a list of times based on a segmentlist
    A time ``t`` will be vetoed if ``start <= t <= end`` for any veto
    segment in the list.
    
    Parameters
    ----------
    times : array
        the times of event triggers to veto
    segmentlist : gwpy.SegmentList
        the list of veto segments to use
   
    Returns
    -------
    keep_args: array
        the arguments of the orignal times array that are not vetoed
    """
    
    # find args that sort times and create sorted times array
    sorted_args = np.argsort(times)
    sorted_times = times[sorted_args]
    
    # initiate array of args to keep; refers to original args of unsorted times array
    keep_args = []
    
    # initiate loop variables; extract first segment 
    j = 0
    a, b = segmentlist[j]
    i = 0
   
    while i < sorted_times.size:
        t = sorted_times[i]
        
        # if before start, not in vetoed segment; move to next trigger now
        if t < a:
            
            # original arg is the ith sorted arg
            original_arg = sorted_args[i]
            keep_args.append(original_arg)
            i += 1
            continue
            
        # if after end, find the next segment and check this trigger again
        if t > b:
            j += 1
            try:
                a, b = segmentlist[j]
                continue
            except IndexError:
                break
                
        # otherwise it must be in veto segment; move on to next trigger
        i += 1
        
    return keep_args
