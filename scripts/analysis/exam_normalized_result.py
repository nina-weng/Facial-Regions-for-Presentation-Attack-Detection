"""
exam the distribution of normalized result
"""
from det_curve import read_tar_non
import matplotlib.pyplot as plt
import numpy as np

f_name = 'normalized_20210623223542.txt'
file_path = '../../results/scores_rec/single_region/'+f_name

bp_score,ap_score = read_tar_non(file_path)

print('BP:\tmean:{}\tvar:{}'.format(np.mean(bp_score),np.std(bp_score)))
print('AP:\tmean:{}\tvar:{}'.format(np.mean(ap_score),np.std(ap_score)))

plt.figure(figsize=(6,6))
plt.hist(ap_score,bins=np.arange(0,1,0.005),color='lightcoral',alpha=0.5,label='ap_score',density=True)
plt.hist(bp_score,bins=np.arange(0,1,0.005),color='lightgreen',alpha=0.5,label='bp_score',density=True)
plt.ylim(0,0.5)
plt.legend()
plt.show()
