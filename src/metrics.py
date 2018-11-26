import numpy as np
import sklearn.metrics
import torch as tr


def average_precision(gtrs_vox, pred_vox):
    N = gtrs_vox.shape[0]
    assert N == pred_vox.shape[0]
    precisions = []
    for i in range(N):
        gtrs_voxel = gtrs_vox[i, ...].flatten()
        pred_voxel = pred_vox[i, ...].flatten()
        precisions.append(
            sklearn.metrics.average_precision_score(
                gtrs_voxel, pred_voxel))

    avg_p = np.array(precisions).mean()
    return avg_p


def iou_t(gtrs, pred, threshold=0.5):
    gtrs = (gtrs > threshold).view(gtrs.shape[0], 32 * 32 * 32)
    pred = (pred > threshold).view(gtrs.shape[0], 32 * 32 * 32)
    union = tr.sum((gtrs | pred).int(), dim=1).float()
    inter = tr.sum((gtrs & pred).int(), dim=1).float()
    return inter / union


def maxIoU(gtrs_vox, pred_vox, step=1e-1):
    ts = np.arange(0., 1., step)
    ious = []
    for t in ts:
        iou = iou_t(gtrs_vox, pred_vox, threshold=t)[None, :]
        ious.append(iou)

    ious = tr.cat(ious, axis=0)
    return tr.mean(tr.max(ious, dim=0))
