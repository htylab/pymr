import nibabel
import numpy as np
from ..heart import ahaseg
import matplotlib.pyplot as plt
from os.path import join, basename

def get_loc(num, got_apex):
    loc = dict()
    if got_apex> 0:
        #got_apex = True    
        loc[3] = [1]*1 + [2]*1 + [3]*1
        loc[4] = [1]*1 + [2]*2 + [3]*1
        loc[5] = [1]*2 + [2]*2 + [3]*1
        loc[6] = [1]*2 + [2]*2 + [3]*2
        loc[7] = [1]*2 + [2]*3 + [3]*2
        loc[8] = [1]*3 + [2]*3 + [3]*2
        loc[9] = [1]*3 + [2]*3 + [3]*3
        loc[10] = [1]*3 + [2]*4 + [3]*3
        loc[11] = [1]*4 + [2]*4 + [3]*3
        loc[12] = [1]*4 + [2]*4 + [3]*4
        loc[13] = [1]*4 + [2]*5 + [3]*4
        loc[14] = [1]*5 + [2]*5 + [3]*4
        loc[15] = [1]*5 + [2]*5 + [3]*5
        loc[16] = [1]*5 + [2]*6 + [3]*5
        loc[17] = [1]*6 + [2]*6 + [3]*5
        loc[18] = [1]*6 + [2]*6 + [3]*6
        loc[19] = [1]*6 + [2]*7 + [3]*6
        loc[20] = [1]*7 + [2]*7 + [3]*6
    else:
        loc[3] = [1]*1 + [2]*1 + [3]*1
        loc[4] = [1]*1 + [2]*1 + [3]*1 + [4]*1
        loc[5] = [1]*1 + [2]*2 + [3]*1 + [4]*1
        loc[6] = [1]*2 + [2]*2 + [3]*1 + [4]*1
        loc[7] = [1]*2 + [2]*2 + [3]*2 + [4]*1
        loc[8] = [1]*2 + [2]*3 + [3]*2 + [4]*1
        loc[9] = [1]*3 + [2]*3 + [3]*2 + [4]*1
        loc[10] = [1]*3 + [2]*3 + [3]*3 + [4]*1
        loc[11] = [1]*3 + [2]*3 + [3]*3 + [4]*2
        loc[12] = [1]*3 + [2]*4 + [3]*3 + [4]*2
        loc[13] = [1]*4 + [2]*4 + [3]*3 + [4]*2
        loc[14] = [1]*4 + [2]*4 + [3]*4 + [4]*2
        loc[15] = [1]*4 + [2]*4 + [3]*4 + [4]*3
        loc[16] = [1]*4 + [2]*5 + [3]*4 + [4]*3
        loc[17] = [1]*5 + [2]*5 + [3]*4 + [4]*3
        loc[18] = [1]*5 + [2]*5 + [3]*5 + [4]*3
        loc[19] = [1]*5 + [2]*6 + [3]*5 + [4]*3
        loc[20] = [1]*6 + [2]*6 + [3]*5 + [4]*3
    
    return loc[num]

def get_slice_label(heart_xyzt):
    #basal: 1, 5
    #mid: 2
    #apical: 3
    #apex: 4
    # 4 and 5 are slices without RV mask
    heart = heart_xyzt.copy()
    curve = np.zeros((heart.shape[2],))
    sys_frame, dia_frame = pyheart.get_frame(heart)
    for slicen in range(heart.shape[2]):    
        heart_mask_xyt = heart[:, :, slicen, :]
        if np.unique(heart_mask_xyt[..., sys_frame]).sum() + np.unique(heart_mask_xyt[..., dia_frame]).sum() == 12:
            curve[slicen] = 1

    curve_and = curve.astype(np.int)
    curve_or = (np.sum(heart, axis=(0, 1, 3)) > 0).astype(np.int)
    curve_diff = curve_or - curve_and
    diff = np.diff(np.sum(heart==1, axis=(0, 1, 3))) * curve_and[1:]
    diff = np.median(diff[curve_and[1:] > 0])

    slice_label = curve_and * 0
    mid_point = curve_and.size//2

    curve_apex = curve_diff.copy()
    curve_basal = curve_diff.copy()

    if diff < 0:
        basal_first = True

        curve_apex[:mid_point] = 0
        curve_basal[mid_point:] = 0
    else:
        basal_first = False
        curve_apex[mid_point:] = 0
        curve_basal[:mid_point] = 0

    got_apex = np.sum(curve_apex) > 0

    if basal_first:
        slice_label[curve_and > 0] = get_loc(np.sum(curve_and), got_apex)
        slice_label[curve_apex > 0] = 4
        slice_label[curve_basal > 0] = 5
    else:
        slice_label[curve_and > 0] = get_loc(np.sum(curve_and), got_apex)[::-1]
        slice_label[curve_apex > 0] = 4
        slice_label[curve_basal > 0] = 5
    
    return slice_label



def convert(f, result_dir=None):

    if isinstance(f, str):
        temp = nibabel.load(f)
        heart = temp.get_fdata()
        affine = temp.affine
        file_input = True
    else:
        heart = f.copy()
        file_input = False
    reverse = np.median(np.diff(np.sum(heart, axis=(0,1,3)))) > 0
    if reverse:
        heart = heart[:,:, ::-1, :]    
    slice_label = get_slice_label(heart)
    #print(slice_label)
    offset = dict()
    offset[1] = 0
    offset[2] = 6
    offset[3] = 12

    count = -1
    heart_aha17_4d = heart  * 0
    for ii in range(slice_label.size):
        #print(ii)
        #print(slice_label[ii])
        count = count + 1
        if (slice_label[ii] == 0) or (slice_label[ii] == 5):
            continue

        if slice_label[ii] == 4:
            temp = heart[:, :, ii, :].copy()
            temp[temp==2] = 17
            temp[temp==1] = 0
            temp[temp==3] = 0
            heart_aha17_4d[:, :, ii, :] = temp
            continue


        heart_xyt = heart[:, :, ii, :].copy()

        curve = np.sum(heart_xyt, axis=(0, 1))
        dia_frame = np.argmax(curve)
        curve[curve==0] = 1e20
        sys_frame = np.argmin(curve)
        #print(dia_frame, sys_frame)
        heart_xy_dia = heart_xyt[..., dia_frame]
        heart_xy_sys = heart_xyt[..., sys_frame]
        if slice_label[ii] == 3:
            nseg = 4
        else:
            nseg = 6

        if (np.sum(heart_xy_dia==3) < 5) or (np.sum(heart_xy_sys==3) < 5):
            slice_label[ii] = 5
            continue


        dia_seg = ahaseg.get_seg((heart_xy_dia==1, heart_xy_dia==2, heart_xy_dia==3), nseg)
        sys_seg = ahaseg.get_seg((heart_xy_sys==1, heart_xy_sys==2, heart_xy_sys==3), nseg)

        dia_seg[dia_seg > 0] = dia_seg[dia_seg > 0] + offset[slice_label[ii]]
        sys_seg[sys_seg > 0] = sys_seg[sys_seg > 0] + offset[slice_label[ii]]

        heart_aha17_4d[:, :, ii, dia_frame] = dia_seg
        heart_aha17_4d[:, :, ii, sys_frame] = sys_seg

    #print('reverse:', reverse)
    if reverse:
        heart_aha17_4d = heart_aha17_4d[:,:, ::-1, :]

    if (result_dir is not None) and file_input:        
        result_f = join(result_dir, basename(f))        
        nii_label = nibabel.Nifti1Image(heart_aha17_4d.astype(np.uint8), affine)
        nibabel.save(nii_label, result_f)
    
    return heart_aha17_4d, sys_frame, dia_frame

def auto_crop(heart_xyzt, crop=None):
    if crop is None:
        heart_xyz = np.sum(heart_xyzt, axis=-1)
        xx, yy, zz = np.nonzero(heart_xyz)
        minx, maxx = max(0, np.min(xx) - 10), min(heart_xyz.shape[0], np.max(xx) + 10)
        miny, maxy = max(0, np.min(yy) - 10), min(heart_xyz.shape[1], np.max(yy) + 10)
        minz, maxz = np.min(zz), np.max(zz)
    else:
        minx, maxx, miny, maxy, minz, maxz = crop
    heart_crop = heart_xyzt[minx:maxx, miny:maxy, minz:maxz, :]
    return heart_crop, (minx, maxx, miny, maxy, minz, maxz)

def get_frame(heart_mask):
    shape = len(heart_mask.shape)
    if shape == 3: #xyt
        curve = np.sum(heart_mask==1, axis=(0, 1))
    elif shape == 4: #xyzt
        curve = np.sum(heart_mask==1, axis=(0, 1, 2))
    else:
        sys_frame = -1
        dia_frame = -1
        return sys_frame, dia_frame

    curve = curve.astype(np.float)
    dia_frame = np.argmax(curve)
    curve[curve==0] = 1e20
    sys_frame = np.argmin(curve)

    return sys_frame, dia_frame
    

