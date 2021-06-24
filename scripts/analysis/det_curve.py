
from DET import DET
import os
import numpy as np
import operator

"""
functions {calculate_roc,get_fmr_op,get_eer} are from tutorials provided by course 02238 
"""
def calculate_roc(gscores, iscores, ds_scores=False, rates=True):
    if isinstance(gscores, list):
        gscores = np.array(gscores, dtype=np.float64)

    if isinstance(iscores, list):
        iscores = np.array(iscores, dtype=np.float64)

    if gscores.dtype == np.int:
        gscores = np.float64(gscores)

    if iscores.dtype == np.int:
        iscores = np.float64(iscores)

    if ds_scores:
        gscores = gscores * -1
        iscores = iscores * -1

    gscores_number = len(gscores)
    iscores_number = len(iscores)

    gscores = zip(gscores, [1] * gscores_number)
    iscores = zip(iscores, [0] * iscores_number)

    gscores = list(gscores)
    iscores = list(iscores)

    scores = np.array(sorted(gscores + iscores, key=operator.itemgetter(0)))
    cumul = np.cumsum(scores[:, 1])

    thresholds, u_indices = np.unique(scores[:, 0], return_index=True)

    fnm = cumul[u_indices] - scores[u_indices][:, 1]
    fm = iscores_number - (u_indices - fnm)

    if rates:
        fnm_rates = fnm / gscores_number
        fm_rates = fm / iscores_number
    else:
        fnm_rates = fnm
        fm_rates = fm

    if ds_scores:
        return thresholds * -1, fm_rates, fnm_rates

    return thresholds, fm_rates, fnm_rates

def get_fmr_op(fmr, fnmr, op):
    index = np.argmin(abs(fmr - op))
    return fnmr[index]

def get_eer(fmr, fnmr):
    diff = fmr - fnmr
    t2 = np.where(diff <= 0)[0]

    if len(t2) > 0:
        t2 = t2[0]
    else:
        return 0, 1, 1, 1

    t1 = t2 - 1 if diff[t2] != 0 and t2 != 0 else t2

    return (fnmr[t2] + fmr[t2]) / 2


def choose_epoch(str_list,criteria='test_loss'):
    num_epoch_exam = len(str_list[2:])
    print('number of epochs need exam:{}'.format(num_epoch_exam))

    # epoch info
    train_accu = np.zeros(num_epoch_exam)
    train_loss = np.zeros(num_epoch_exam)
    test_accu = np.zeros(num_epoch_exam)
    test_loss = np.zeros(num_epoch_exam)
    APCER = np.zeros(num_epoch_exam)
    BPCER = np.zeros(num_epoch_exam)

    for idx,epoch_str in enumerate(str_list[2:]):
        test_data = epoch_str.split('train accuracy:')[0]
        epoch_info = epoch_str.split('train accuracy:')[1]

        # collect epoch info
        train_accu_this = float(epoch_info.split('train loss:')[0])
        train_loss_this = float(epoch_info.split('train loss:')[1].split('\n')[0])
        test_accu_this = float(epoch_info.split('test accuracy:')[1].split('\t')[0])
        test_loss_this = float(epoch_info.split('test loss:')[1].split('\n')[0])
        APCER_this = float(epoch_info.split('APCER:')[1].split('\t')[0])
        BPCER_this = float(epoch_info.split('BPCER:')[1].split('\n')[0])

        train_accu[idx] = train_accu_this
        train_loss[idx] = train_loss_this
        test_accu[idx] = test_accu_this
        test_loss[idx] = test_loss_this
        APCER[idx] = APCER_this
        BPCER[idx] = BPCER_this

    if criteria == 'train_loss':
        best_epoch = np.argmin(train_loss)
    elif criteria == 'test_loss':
        best_epoch = np.argmin(test_loss)
    else:
        raise Exception('criteria {} not implemented'.format(criteria))

    print('the best epoch is {}, based on the criteria {}'.format(best_epoch,criteria))
    print('epoch {}: train accu:{}\ttrain_loss:{}\ttest_accu:{}\ttest_loss:{}\tAPCER:{}\tBPCER:{}'.format(best_epoch,
            train_accu[best_epoch],train_loss[best_epoch],test_accu[best_epoch],test_loss[best_epoch],
            APCER[best_epoch],BPCER[best_epoch]))
    return str_list[best_epoch]



def read_tar_non(txt_fpath,file_type='single'):
    scores_bp = []
    scores_ap = []

    with open(txt_fpath,'r') as f:
        txt_data = f.read()

    epochs_data = txt_data.split('epoch')

    if file_type=='fusion':
        chosen_epoch = choose_epoch(epochs_data)
    else:
        chosen_epoch = epochs_data[-1]

    contents = chosen_epoch.split('\n')

    for i in range(1,len(contents)):
        if len(contents[i].split(',')) != 3:
            # print('WARNING:{}\t{}'.format(i,contents[i]))
            continue

        s,s_,label = contents[i].split(',')
        if label[0] == '[':
            label = label[1:-1]
        if label == '0':
            scores_bp.append(float(s))
        elif label == '1':
            scores_ap.append(float(s))
        else:
            raise Exception('label error')

    return np.array(scores_bp),np.array(scores_ap)


if __name__ == '__main__':
    file_type = 'fusion'
    NUM_RF = 3


    if file_type == 'single':
        scores_rec_dir = '../../results/scores_rec/single_region'
    elif file_type == 'fusion':
        scores_rec_dir = '../../results/scores_rec/region_fusion'
    else:
        raise Exception('file type {} not implemented'.format(file_type))

    face_regions=[]
    tars = []
    nons = []

    for fname in os.listdir(scores_rec_dir):
        # print(fname.split('202106')[1])
        if fname.split('202106')[1][:2] == '23' or fname.split('202106')[1][:2] == '24':
            # print(fname)
            if file_type == 'single':
                fr_name = fname.split('_2021')[0]
            elif file_type == 'fusion':
                fr_name = fname.split('_2021')[0].split('fusion_')[1]
                num_fr = len(fr_name.split('-'))
                if NUM_RF!= None:
                    if NUM_RF != num_fr:
                        continue

            face_regions.append(fr_name)
            tar,non = read_tar_non(os.path.join(scores_rec_dir,fname),file_type=file_type)
            tars.append(tar)
            nons.append(non)
            # calculate the EER
            thresholds, apcer, bpcer = calculate_roc(tar, non)
            eer = get_eer(apcer, bpcer)*100
            print('EER for region {}:{:.2f}%'.format(fr_name,eer))


    print('includes {} region result'.format(len(face_regions)))

    det = DET(biometric_evaluation_type='PAD', plot_title='APCER-BPCER', abbreviate_axes=True, plot_eer_line=True,
            figsize=(9,5))

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

