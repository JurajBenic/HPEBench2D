import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd

def L1(gt: np.ndarray, pp: np.ndarray, vi: np.ndarray = None) -> float:
    """
        gt - ground truth array Nx2
        pp - predictet point array Nx2
        
        L1 norma: 1/n * sum_{i=1}^n( |y_i - y_i^p| )
        
        n = broj tocaka
        y_i = i-ta ground truth tocka s koordinatama x,y
        y_i^p = i-ta predictet point tocka s koordinatama x,y

        Primjer za svaki-sa-svakim
        L1(gt[:, None, ...], pp[None, :, ...])
    """
    if vi is None:
        return np.mean(np.linalg.norm(gt - pp, 1, axis=-1), axis=-1)
    
    vi_1 = 1. * (vi > 0)
    return np.sum(np.linalg.norm(gt - pp, 1, axis=-1) * vi_1, axis=-1) / np.sum(vi_1, axis=-1)

def L2(gt: np.ndarray, pp: np.ndarray, vi: np.ndarray = None) -> float:
    """
        gt - ground truth array Nx2
        pp - predictet point array Nx2
        
        L1 norma: 1/n * sum_{i=1}^n( (y_i - y_i^p)^2 )
        
        n = broj tocaka
        y_i = i-ta ground truth tocka s koordinatama x,y
        y_i^p = i-ta predictet point tocka s koordinatama x,y
    """
    if vi is None:
        return np.mean(np.linalg.norm(gt - pp, axis=-1), axis=-1)
    
    vi_1 = 1. * (vi > 0)
    return np.sum(np.linalg.norm(gt - pp, axis=-1) * vi_1, axis=-1) / np.sum(vi_1, axis=-1)

def PDJ(gt: np.ndarray, pp: np.ndarray, bb_diagonal: float, percet_of_bb: float) -> float:
    """
    Percentage of Detected Joints (PDJ)
        gt - ground truth array Nx2
        pp - predictet point array Nx2
    """
    di = np.linalg.norm(gt - pp, axis=-1)
    return np.mean(di<bb_diagonal*percet_of_bb, axis=-1)

    
def OKS(gt: np.ndarray, pp: np.ndarray, s: float = 1000, vi: np.ndarray = None, ki: np.ndarray = None) -> float:
    """
        gt - ground truth array Nx2
        pp - predictet point array Nx2
        s - is the human scale: the square root of the object segment area
            an be approximated by heuristically multiplying a scaling factor of 0.53 with the bounding box area
        ki - per-keypoint constant that controls fall off, dana u coco
        
        OKS = sum( exp(-di^2/(2*s*ki^2)) )/ len(ki)
        
        di= euklidska udaljesnot cvorova
    """
    if ki is None:
        # COCO
        # constant for keypoint i
        ki = 2*np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    
    di = np.linalg.norm(gt - pp, axis=-1)
    KSi = np.exp( (-di**2)/(2*s*(ki[...,:]**2)) )
    
    if vi is None:
        return np.mean(KSi, axis=-1)
    
    vi_1 = 1. * (vi > 0)
    return np.sum(KSi * vi_1, axis=-1) / np.sum(vi_1, axis=-1)


def assignPeople(gts: np.ndarray, pps: np.ndarray, vi: np.ndarray, si: np.ndarray, **kwargs)-> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if gts.size == 0 or pps.size == 0:
        return (np.array([]), np.array([]), np.array([]))
    costs = OKS(gts[:,None,...], pps[None,:,...], vi=vi[:,None,:], s=si[:,None,None])
    row_ind, col_ind = linear_sum_assignment(costs, maximize=True)
    indx = np.argsort(-costs[row_ind, col_ind])
    row_ind = row_ind[indx]
    col_ind = col_ind[indx]
    return (row_ind, col_ind, costs[row_ind, col_ind])

def assignPeopleGreedy(gts: np.ndarray, pps: np.ndarray, vi: np.ndarray, si: np.ndarray, **kwargs)-> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if gts.size == 0 or pps.size == 0:
        return (np.array([]), np.array([]), np.array([]))
    costs = OKS(gts[:,None,...], pps[None,:,...], vi=vi[:,None,:], s=si[:,None,None])
    row_ind = []
    col_ind = []
    rows = [-1] * costs.shape[0]
    cols = [-1] * costs.shape[1]
    for ij in np.argsort(-costs.ravel()):
        (i, j) = np.unravel_index(ij, costs.shape)
        if rows[i] == -1 and cols[j] == -1:
            rows[i] = j
            cols[j] = i
            row_ind.append(i)
            col_ind.append(j)
    
    row_ind = np.array(row_ind)
    col_ind = np.array(col_ind)
    return (row_ind, col_ind, costs[row_ind, col_ind])

def assignPeopleL2(gts: np.ndarray, pps: np.ndarray, vi: np.ndarray, **kwargs)-> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if gts.size == 0 or pps.size == 0:
        return (np.array([]), np.array([]), np.array([]))
    costs = L2(gts[:,None,...], pps[None,:,...], vi[:,None,:])
    costs_og = costs.copy()
    row_ind = []
    col_ind = []
    while True:
        (i, j) = np.unravel_index(np.argmin(costs), costs.shape)
        if costs[i,j] == np.inf: break
        row_ind.append(i)
        col_ind.append(j)
        costs[i,:] = np.inf
        costs[:,j] = np.inf
    row_ind = np.array(row_ind)
    col_ind = np.array(col_ind)
    return (row_ind, col_ind, costs_og[row_ind, col_ind])

    
if __name__ == '__main__':
    import json
    import re
    import os
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose',
                        action='store_true')
    parser.add_argument('-s', '--save',
                        action='store_true')
    

    args = parser.parse_args()
    
    pth = '/mnt/hdd/datasets/IoTGym/panoptic/'
    datasets = [
        '160906_band2',
        '160906_ian1',
        '160906_pizza1',
        '170221_haggling_m3',
        '170307_dance5',
        '170915_office1',
        '171026_pose3'
        ]

    methods = [
        'openpose',
        'rtmo',
        'rtmpose-l',
        'yolo11x',
        'sapiens'
        ]
    
    regx = re.compile(r'hdPose2D_(.*)\.json')
    
    method_results_dict = {
        'dataset': [],
        'video': [],
        'method': [],
        'frame': [],
        'idGt': [],
        'idPs': [],
        'L1': [],
        'L2': [],
        'OKS': [],
#                        'PDJ': [],
    }
    
    for dat in datasets:
        dat_path = os.path.join(pth, dat)

        for fl in os.listdir(dat_path):
            if match := regx.match(fl):
                fl_name = match[1]
                fl_path = os.path.join(dat_path, fl)
                with open(fl_path) as fl_json:
                    data_gt = json.load(fl_json)

                #plot_data = { method: {'scores':[], 'L2': []} for method in methods }

                for method in methods:
                    meth_path = os.path.join(pth, 'results', method, f'{method}_{dat}_hd_{fl_name}.json')
                    with open(meth_path) as mt_json:
                        data_dt = json.load(mt_json)


                    for (frame_id, frame_data) in sorted(data_gt.items(), key=lambda x: int(x[0])):
                        person_ids_gt = []
                        keypoints_gt = []
                        visibility_gt = []
                        bboxs_gt = []
                        for person in frame_data['bodies']:
                            if sum(person['keypoint_scores']) == 0.:
                                continue
                            person_ids_gt.append(person['id'])
                            keypoints_gt.append(person['keypoints'])
                            visibility_gt.append(person['keypoint_scores'])
                            kp = np.array(person['keypoints'])
                            vs = np.array(person['keypoint_scores'])
                            kp = kp[vs > 0]
                            bboxs_gt.append( np.prod(kp.max(axis=0) - kp.min(axis=0)) )
                                          
                        person_ids_gt = np.array(person_ids_gt)
                        keypoints_gt = np.array(keypoints_gt)
                        visibility_gt = np.array(visibility_gt)
                        bboxs_gt = np.maximum(0.53 * np.array(bboxs_gt), 1e-20)

                        
                        person_ids_dt = []
                        keypoints_dt = []
                        scores_dt = []
                        for person in data_dt[frame_id]['bodies']:
                            person_ids_dt.append(person['id'])
                            keypoints_dt.append(person['keypoints'])
                            scores_dt.append(person['keypoint_scores'])
                        
                            
                    
                        person_ids_dt = np.array(person_ids_dt)
                        keypoints_dt = np.array(keypoints_dt)
                        scores_dt = np.array(scores_dt)

                        try:
     
                            row_gri, col_gri, costs_OKS = assignPeople(keypoints_gt, keypoints_dt, visibility_gt, bboxs_gt)
                            #if row_gri.size > 0:
                            #    dists = np.linalg.norm(keypoints_gt[row_gri] - keypoints_dt[col_gri], axis=-1)**2 / bboxs_gt[row_gri][:,None]
                                #Y = dists[visibility_gt[row_gri] > 0]
                                #X = scores_dt[col_gri][visibility_gt[row_gri] > 0]
                                
                                #plot_data[method]['scores'].extend(X)
                                #plot_data[method]['L2'].extend(Y)
                            
                            
                            if args.verbose:
                                print(f'{dat = } {method = } {frame_id = } {costs_OKS = }')

                            if row_gri.size > 0:
                                gts = keypoints_gt[row_gri]
                                pps = keypoints_dt[col_gri]
                                vis = visibility_gt[row_gri]

                                costs_L1 = L1(gts, pps, vis)
                                costs_L2 = L2(gts, pps, vis)

                                for idGt, idPs, l1, l2, oks in zip(person_ids_gt[row_gri],
                                                                   person_ids_dt[col_gri],
                                                                   costs_L1,
                                                                   costs_L2,
                                                                   costs_OKS):
                            
                                    method_results_dict['dataset'].append(dat)
                                    method_results_dict['method'].append(method)
                                    method_results_dict['frame'].append(frame_id)
                                    method_results_dict['video'].append(fl_name)
                                    method_results_dict['idGt'].append(idGt)
                                    method_results_dict['idPs'].append(idPs)
                                    method_results_dict['L1'].append(l1)
                                    method_results_dict['L2'].append(l2)
                                    method_results_dict['OKS'].append(oks)
                                

                            row_set = set(person_ids_gt[row_gri] if row_gri.size > 0 else [])
                            for idGt in person_ids_gt:
                            
                                if not idGt in row_set:
                                    method_results_dict['dataset'].append(dat)
                                    method_results_dict['method'].append(method)
                                    method_results_dict['frame'].append(frame_id)
                                    method_results_dict['video'].append(fl_name)
                                    method_results_dict['idGt'].append(idGt)
                                    method_results_dict['idPs'].append('-1')
                                    method_results_dict['L1'].append(-1)
                                    method_results_dict['L2'].append(-1)
                                    method_results_dict['OKS'].append(-1)


                            col_set = set(person_ids_dt[col_gri] if col_gri.size > 0 else [])
                            for idDt in person_ids_dt:
                            
                                if not idDt in col_set:
                                    method_results_dict['dataset'].append(dat)
                                    method_results_dict['method'].append(method)
                                    method_results_dict['frame'].append(frame_id)
                                    method_results_dict['video'].append(fl_name)
                                    method_results_dict['idGt'].append('-1')
                                    method_results_dict['idPs'].append(idDt)
                                    method_results_dict['L1'].append(-1)
                                    method_results_dict['L2'].append(-1)
                                    method_results_dict['OKS'].append(-1)
                                
                                
                        except Exception as e:
                            print(e)
                            print(f'{dat = } {fl_name = } {method = } {frame_id = }')
                            raise e


                
                #for (method, values) in plot_data.items():
                #    plt.semilogy(values['scores'], values['L2'], '.', label=method)
                #plt.legend()
                #plt.xlabel('Confidence score')
                #plt.ylabel('L2^2/bbox')
                #plt.title(f'{dat}_{fl_name}')
                #plt.savefig(f'./figs/{dat}_{fl_name}.pdf')
                #plt.close()
                

    results_csv = pd.DataFrame.from_dict(method_results_dict)
    if args.save:
        results_csv.to_csv(os.path.join(pth, 'results/', 'results.csv'))
