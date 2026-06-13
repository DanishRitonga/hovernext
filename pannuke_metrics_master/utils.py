import numpy as np
from scipy.optimize import linear_sum_assignment

####
def get_fast_pq(true, pred, match_iou=0.5):
    """Compute fast PQ between GT and prediction instance maps.

    Both arrays are remapped to contiguous IDs internally.
    Returns [dq, sq, pq] and pairing info.
    """
    assert match_iou >= 0.0

    true = remap_label(np.copy(true))
    pred = remap_label(np.copy(pred))

    true_ids = [int(x) for x in np.unique(true)]
    pred_ids = [int(x) for x in np.unique(pred)]
    true_ids_fg = [x for x in true_ids if x != 0]
    pred_ids_fg = [x for x in pred_ids if x != 0]

    n_true = len(true_ids_fg)
    n_pred = len(pred_ids_fg)

    if n_true == 0 or n_pred == 0:
        tp = 0
        fp = n_pred
        fn = n_true
        dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
        return [dq, 0.0, 0.0], [[], [], list(true_ids_fg), list(pred_ids_fg)]

    true_masks = {tid: np.array(true == tid, np.uint8) for tid in true_ids_fg}
    pred_masks = {pid: np.array(pred == pid, np.uint8) for pid in pred_ids_fg}

    pairwise_iou = np.zeros([n_true, n_pred], dtype=np.float64)

    for ti, tid in enumerate(true_ids_fg):
        t_mask = true_masks[tid]
        overlap_ids = np.unique(pred[t_mask > 0])
        for pid in overlap_ids:
            pid = int(pid)
            if pid == 0 or pid not in pred_masks:
                continue
            pi = pred_ids_fg.index(pid)
            p_mask = pred_masks[pid]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_iou[ti, pi] = inter / (total - inter)

    if match_iou >= 0.5:
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true = list(paired_true + 1)
        paired_pred = list(paired_pred + 1)
    else:
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        mask = paired_iou > match_iou
        paired_true = list(paired_true[mask] + 1)
        paired_pred = list(paired_pred[mask] + 1)
        paired_iou = paired_iou[mask]

    unpaired_true = [i + 1 for i in range(n_true) if (i + 1) not in paired_true]
    unpaired_pred = [i + 1 for i in range(n_pred) if (i + 1) not in paired_pred]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
    sq = paired_iou.sum() / (tp + 1.0e-6) if tp > 0 else 0.0

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]
#####

def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1    
    return new_pred
####


def binarize(x):
    '''
    convert multichannel (multiclass) instance segmetation tensor
    to binary instance segmentation (bg and nuclei),

    :param x: B*B*C (for PanNuke 256*256*5 )
    :return: Instance segmentation
    '''
    out = np.zeros([x.shape[0], x.shape[1]])
    count = 1
    for i in range(x.shape[2]):
        x_ch = x[:,:,i]
        unique_vals = np.unique(x_ch)
        unique_vals = unique_vals.tolist()
        if 0 in unique_vals:
            unique_vals.remove(0)
        for j in unique_vals:
            x_tmp = x_ch == j
            x_tmp_c = 1- x_tmp
            out *= x_tmp_c
            out += count*x_tmp
            count += 1
    out = out.astype('int32')
    return out
####

def get_tissue_idx(tissue_indices, idx):
    for i in range(len(tissue_indices)):
        if tissue_indices[i].count(idx) == 1:
            tiss_idx = i
    return tiss_idx 