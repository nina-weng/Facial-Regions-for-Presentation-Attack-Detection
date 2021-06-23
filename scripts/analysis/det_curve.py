
from DET import DET
import os
import numpy as np

def read_tar_non(txt_fpath):
    scores_bp = []
    scores_ap = []

    with open(txt_fpath,'r') as f:
        txt_data = f.read()

    epochs_data = txt_data.split('epoch')
    last_epoch = epochs_data[-1]

    contents = last_epoch.split('\n')

    for i in range(1,len(contents)):
        if len(contents[i].split(',')) != 3:
            print('WARNING:{}\t{}'.format(i,contents[i]))
            continue

        s,s_,label = contents[i].split(',')
        if label == '0':
            scores_bp.append(float(s))
        elif label == '1':
            scores_ap.append(float(s))
        else:
            raise Exception('label error')

    return np.array(scores_bp),np.array(scores_ap)


if __name__ == '__main__':
    scores_rec_dir = '../../results/scores_rec/single_region'

    face_regions=[]
    tars = []
    nons = []

    for fname in os.listdir(scores_rec_dir):
        # print(fname.split('202106')[1])
        if fname.split('202106')[1][:2] == '23':
            face_regions.append(fname.split('_2021')[0])
            tar,non = read_tar_non(os.path.join(scores_rec_dir,fname))
            tars.append(tar)
            nons.append(non)

    print('includes {} region result'.format(len(face_regions)))

    det = DET(biometric_evaluation_type='PAD', plot_title='APCER-BPCER', abbreviate_axes=True, plot_eer_line=True)

    det.x_limits = np.array([1e-3, 5e-1])
    det.y_limits = np.array([1e-3, 5e-1])
    det.x_ticks = np.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2,5e-2,1e-1,2e-1,4e-1])
    det.x_ticklabels = np.array([ '0.1', '0.2', '0.5', '1','2','5','10','20','40'])
    det.y_ticks = np.array([ 1e-3, 2e-3, 5e-3, 1e-2,2e-2,5e-2,1e-1,2e-1,4e-1])
    det.y_ticklabels = np.array(['0.1', '0.2', '0.5', '1','2','5','10','20','40'])


    det.create_figure()
    for i in range(len(face_regions)):
        det.plot(tar=tars[i], non=nons[i], label=face_regions[i])



    det.legend_on(bbox_to_anchor=(1.05, 1), loc='upper left')
    det.show()
    # det.save('../../results/plots/face_regions_det','png')

