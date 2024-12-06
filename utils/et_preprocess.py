import numpy as np

def simple_mad(angular_vel, thresh = 5):
    # Check input format 
    if len(angular_vel.shape) == 1:
        angular_vel = angular_vel[:,None]
    
    # Median of input 
    median = np.median(angular_vel)
    
    # Absolute difference of input values from median 
    diff = (angular_vel - median)**2
    diff = np.sqrt(diff)
    
    # Median of absolute difference 
    med_abs_deviation = np.median(diff)
    
    # New threshold is median + coefficient * median of absolute difference 
    saccade_thresh = median + thresh*med_abs_deviation
    return saccade_thresh

def at_mad(angular_vel, th_0=200):
    threshs = []
    thresh_coeff = 3*1.48
    
    # Check input format 
    if len(angular_vel.shape) == 1:
        angular_vel = angular_vel[:,None]

    # As long as difference between newly calculated threshold and previous threshold is greater 1
    while True:
        threshs.append(th_0) # store current threshold
        angular_vel = angular_vel[angular_vel < th_0] # cut all values above current threshold      
        median = np.median(angular_vel) # median of values that are < threshold
        diff = (angular_vel - median)**2 # absolute difference between values and median
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff) # median of absolute difference 
        th_1 = median + thresh_coeff*med_abs_deviation # new thresh = median + coefficient * median absolute deviation
#         print(th_0, th_1)
        if (th_0 - th_1)>1: # make new thresh current thresh and restart in loop             
            th_0 = th_1
        else: # difference between threshs small enough, new thresh becomes saccade thresh 
            saccade_thresh = th_1 
            threshs.append(saccade_thresh)
            break
    return saccade_thresh



