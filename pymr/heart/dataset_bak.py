import nibabel
import numpy as np
from ..heart import ahaseg
from .cine import get_frame, auto_crop
from .viz import montage
import matplotlib.pyplot as plt
from os.path import join, basename
import glob
import os
from tqdm import tqdm
import shutil

def get_loc(num, got_apex):
    loc = dict()
    if got_apex> 0:
        #got_apex = True
        mid = num//3
        basal = mid
        apical = mid
        if num % 3 == 1:
            mid = mid + 1
        elif num % 3 == 2:
            basal += 1
            mid += 1
        
        loc = [1]*basal + [2]*mid + [3]*apical
        #loc[3] = [1]*1 + [2]*1 + [3]*1
        #loc[4] = [1]*1 + [2]*2 + [3]*1
        #loc[5] = [1]*2 + [2]*2 + [3]*1
        #loc[6] = [1]*2 + [2]*2 + [3]*2
        #loc[7] = [1]*2 + [2]*3 + [3]*2


    else:
        
        apex = num // 7

        num = num - apex

        mid = num//3
        basal = mid
        apical = mid
        if num % 3 == 1:
            mid = mid + 1
        elif num % 3 == 2:
            basal += 1
            mid += 1
            
        loc = [1]*basal + [2]*mid + [3]*apical + [4]*apex

    
    return loc

def get_slice_label(heart_xyzt):
    #basal: 1, 5
    #mid: 2
    #apical: 3
    #apex: 4
    # 4 and 5 are slices without RV mask
    heart = heart_xyzt.copy()
    curve = np.zeros((heart.shape[2],))
    sys_frame, dia_frame = get_frame(heart)
    for slicen in range(heart.shape[2]):    
        heart_mask_xyt = heart[:, :, slicen, :]
        if np.unique(heart_mask_xyt[..., sys_frame]).sum() + np.unique(heart_mask_xyt[..., dia_frame]).sum() == 12:
            curve[slicen] = 1

    curve_and = curve.astype(np.int)
    curve_or = (np.sum(heart, axis=(0, 1, 3)) > 0).astype(np.int)
    curve_diff = curve_or - curve_and
    #diff = np.diff(np.sum(heart==1, axis=(0, 1, 3))) * curve_and[1:]
    #diff = np.median(diff[curve_and[1:] > 0])
    
    temp = np.sum(heart==1, axis=(0, 1, 3))
    temp = temp[temp>0]
    diff = temp[-1] - temp[0]

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
        slice_label[curve_basal > 0] = 11
    else:
        slice_label[curve_and > 0] = get_loc(np.sum(curve_and), got_apex)[::-1]
        slice_label[curve_apex > 0] = 4
        slice_label[curve_basal > 0] = 11
    #print(slice_label)
    return slice_label

def proc_basal_without_RV(mask1, mask5):
    if np.unique(mask5).sum() == 6:
        seg = ahaseg.get_seg(mask5, nseg=6)
        return seg
    
    # find nearest point with ahaseg
    xx5, yy5 = np.nonzero(mask5==2)
    xx1, yy1 = np.nonzero(mask1)
    seg5 = mask5 * 0
    for ii in range(xx5.size):
        index = np.sqrt((xx1 - xx5[ii])**2 + (yy1 - yy5[ii])**2).argmin()
        seg5[xx5[ii], yy5[ii]] = mask1[xx1[index], yy1[index]]
    return seg5


def convert(f, dataset='MMS', result_dir=None):
    if isinstance(f, str):
        temp = nibabel.load(f)
        heart = temp.get_fdata()
        affine = temp.affine
        file_input = True
        new_f = basename(f)
    else:
        heart = f.copy()
        file_input = False        

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
        if (slice_label[ii] == 0) or (slice_label[ii] == 11):
            continue

        if slice_label[ii] == 4:
            temp = heart[:, :, ii, :].copy()
            temp[temp==2] = 17
            temp[temp==1] = 0
            temp[temp==3] = 0
            heart_aha17_4d[:, :, ii, :] = temp
            continue


        heart_xyt = heart[:, :, ii, :].copy()

        sys_frame, dia_frame = get_frame(heart_xyt)

        #print(dia_frame, sys_frame)
        heart_xy_dia = heart_xyt[..., dia_frame]
        heart_xy_sys = heart_xyt[..., sys_frame]
        if slice_label[ii] == 3:
            nseg = 4
        else:
            nseg = 6

        dia_seg = ahaseg.get_seg(heart_xy_dia, nseg)
        sys_seg = ahaseg.get_seg(heart_xy_sys, nseg)
        dia_seg[dia_seg > 0] = dia_seg[dia_seg > 0] + offset[slice_label[ii]]
        sys_seg[sys_seg > 0] = sys_seg[sys_seg > 0] + offset[slice_label[ii]]
        
        if slice_label[ii] == 3:
            if (np.sum(dia_seg) == 0) or (np.sum(sys_seg) == 0):
                dia_seg = heart_xy_dia.copy()
                dia_seg[dia_seg==2] = 17
                dia_seg[dia_seg==1] = 0
                dia_seg[dia_seg==3] = 0
                
                sys_seg = heart_xy_sys.copy()
                sys_seg[sys_seg==2] = 17
                sys_seg[sys_seg==1] = 0
                sys_seg[sys_seg==3] = 0
                
   
                
        

        heart_aha17_4d[:, :, ii, dia_frame] = dia_seg
        heart_aha17_4d[:, :, ii, sys_frame] = sys_seg
    
    
    #=====================process basal
    go = True
    slice_label_temp = slice_label.copy()
    while go:
        if np.any(np.diff(slice_label_temp)==-10):
            index11 = int(np.where(np.diff(slice_label_temp)==-10)[0])
            index1 = index11 + 1
            slice_label_temp[index11] = 1
            go = True

        elif np.any(np.diff(slice_label_temp)==10):
            index11 = int(np.where(np.diff(slice_label_temp)==10)[0] + 1)
            index1 = index11 - 1
            slice_label_temp[index11] = 1
            go = True
        else:
            index11 = -1
            go = False

        if go:
            mask11_sys = heart[:, :, index11, sys_frame]
            mask11_dia = heart[:, :, index11, dia_frame]
            seg1_sys = heart_aha17_4d[:, :, index1, sys_frame]
            seg1_dia = heart_aha17_4d[:, :, index1, dia_frame]

            seg11_dia = proc_basal_without_RV(seg1_dia, mask11_dia)
            seg11_sys = proc_basal_without_RV(seg1_sys, mask11_sys)
            heart_aha17_4d[:, :, index11, sys_frame] = seg11_sys
            heart_aha17_4d[:, :, index11, dia_frame] = seg11_dia
            #plt.figure()
            #plt.imshow(seg11)
            #print(slice_label_temp)  


    heart_aha17_4d[heart==1] = 18
    heart_aha17_4d[heart==3] = 19

    if (result_dir is not None) and file_input:
        if dataset=='MMS':
            new_f = basename(f).replace('_gt', '_lab19')
        else:
            new_f = basename(f).replace('.nii.gz', '_lab19.nii.gz')
        
        result_f = join(result_dir, new_f)        
        nii_label = nibabel.Nifti1Image(heart_aha17_4d.astype(np.uint8), affine)
        nibabel.save(nii_label, result_f)

    return heart_aha17_4d


def safe_mkdir(dirname):
    try:
        os.mkdir(dirname)
    except:
        pass

def convert_mms(source_dir, dest_dir=None):
    
    train_ffs = glob.glob(join(source_dir, 'Training', 'Labeled', '**', '*gt.nii.gz'))
    val_ffs = glob.glob(join(source_dir, 'Validation', '**', '*gt.nii.gz'))
    test_ffs = glob.glob(join(source_dir, 'Testing', '**', '*gt.nii.gz'))
    

    
    
    if dest_dir is None:
        dest_dir = join(source_dir, 'Convert_lab19')
        
        
    train_dir = join(dest_dir, 'Training')
    val_dir = join(dest_dir, 'Validation')
    test_dir = join(dest_dir, 'Testing')
    jpg_dir = join(dest_dir, 'jpg')
    safe_mkdir(dest_dir)
    safe_mkdir(train_dir)
    safe_mkdir(val_dir)
    safe_mkdir(test_dir)
    safe_mkdir(jpg_dir)
    
    loop = [(train_ffs, train_dir), (val_ffs, val_dir), (test_ffs, test_dir)]
    
    for ffs, result_dir in loop:
        print(result_dir)
        for f in ffs:
            try:
                heart_aha17_4d = convert(f, dataset='MMS', result_dir=result_dir)
                sys_frame, dia_frame = get_frame(nibabel.load(f).get_fdata())
                heart_crop, crop = auto_crop(heart_aha17_4d)            
                jpg_file = join(jpg_dir, basename(f) + '.jpg')
                montage(np.concatenate([heart_crop[..., sys_frame], heart_crop[..., dia_frame]], axis=-1), fname=jpg_file, display=False)
                img_file = f.replace('_gt.nii.gz', '.nii.gz')
                shutil.copyfile(img_file, join(result_dir, basename(img_file)))
                shutil.copyfile(f, join(result_dir, basename(f)))

            except Exception as e:

                print(e)
                print('error:', f)
                
                
                
def convert_acdc(source_dir, dest_dir=None):
    
    if dest_dir is None:
        dest_dir = join(source_dir, 'Convert_lab19')
    

    jpg_dir = join(dest_dir, 'jpg')

    safe_mkdir(dest_dir)
    safe_mkdir(jpg_dir)

    for f in glob.glob(join(source_dir, 'training', '**', '*_4d.nii.gz')):    
        #try:
        #print(f)
        base_f = basename(f)
        temp = nibabel.load(f)
        vol4d = temp.get_fdata()
        affine = temp.affine
        heart_temp = vol4d * 0
        for gt in glob.glob(f.replace('_4d.nii.gz','*gt.nii.gz' )):
            gt_mask = nibabel.load(gt).get_fdata()
            frame = int(basename(gt).split('_')[1].replace('frame', ''))
            heart_temp[..., frame-1] = gt_mask

        heart = heart_temp.copy()
        heart[heart_temp==1] = 3
        heart[heart_temp==3] = 1

        heart_aha17_4d = convert(heart)

        result_f = join(dest_dir, base_f.replace('_4d.nii.gz', '_gt.nii.gz'))      
        nii_label = nibabel.Nifti1Image(heart.astype(np.uint8), affine)
        nibabel.save(nii_label, result_f)


        result_f = join(dest_dir, base_f.replace('_4d.nii.gz', '_lab19.nii.gz'))      
        nii_label = nibabel.Nifti1Image(heart_aha17_4d.astype(np.uint8), affine)
        nibabel.save(nii_label, result_f)


        result_f = base_f.replace('_4d.nii.gz', '.nii.gz')
        shutil.copyfile(f, join(dest_dir, result_f))

        sys_frame, dia_frame = get_frame(heart)
        heart_crop, crop = auto_crop(heart_aha17_4d)            
        jpg_file = join(jpg_dir, base_f + '.jpg')
        montage(np.concatenate([heart_crop[..., sys_frame], heart_crop[..., dia_frame]], axis=-1), fname=jpg_file, display=False)

        #except Exception as e:

        #    print(e)
        #    print('error:', f)
