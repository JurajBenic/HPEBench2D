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
    '''
       gts - ground truth keypoints PxNx2
       pps - predicted keypoints    RxNx2
       vi - visibility of keypoints PxN
       si - huma scale needed for function OKS Px1
       
       returns indexes and OKS values of matches (gt_indexs, pp_indexs, oks) in descending order of the match, i.e.
       gt_indexs[i] and pp_indexs[i] are matched ground truth to prediction with OKS value of oks[i].
    '''
    if gts.size == 0 or pps.size == 0:
        return (np.array([]), np.array([]), np.array([]))
    costs = OKS(gts[:,None,...], pps[None,:,...], vi=vi[:,None,:], s=si[:,None,None])
    row_ind, col_ind = linear_sum_assignment(costs, maximize=True)
    indx = np.argsort(-costs[row_ind, col_ind])
    row_ind = row_ind[indx]
    col_ind = col_ind[indx]
    return (row_ind, col_ind, costs[row_ind, col_ind])
    
if __name__ == '__main__':
    import json
    import re
    import os
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', help='Print progress.')
    parser.add_argument('-path', type=str, required=True, help='Path to the folder of results. The structure of the folder has to be similar to the paper one.')
    parser.add_argument('-datasets', type=str, nargs='*', help='Names of datasets/folders to process. If left blank it will process all datasest in the given -path directory.')
    parser.add_argument('-methods', type=str, nargs='*', help='Names of methods/folders to process. If left blank it will process all methods in the given path/dataset directory.')
    parser.add_argument('-out', type=str, help='File to save the results. If left blank then the results will not be saved.')
    
    args = parser.parse_args()
    
    pth = args.path
    datasets = [ x for x in os.listdir(pth) if os.path.isdir(os.path.join(pth, x)) and x != 'results' ] if args.datasets is None else args.datasets
    methods = [ x for x in os.listdir(os.path.join(pth, 'results')) if os.path.isdir(os.path.join(pth, 'results', x)) ] if args.methods is None else args.methods
    
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
                

    results_csv = pd.DataFrame.from_dict(method_results_dict)
    if not args.out is None:
        results_csv.to_csv(args.out)
    else:
        print(results_csv.to_string())
