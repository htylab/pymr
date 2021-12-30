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


def get_cmap():
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    viridis = cm.get_cmap("jet", 21)
    newcolors = viridis(np.linspace(0, 1, 21))
    newcolors[0] = np.array([0, 0, 0, 1])
    newcolors[1] = np.array([0, 0.6, 0.6, 1])
    newcolors[2] = np.array([0.5, 0, 0.5, 1])
    newcolors[3] = np.array([0.8, 0.8, 0.5, 1])
    newcolors[4:21] = viridis(np.linspace(0, 1, 17))
    newcmp = ListedColormap(newcolors)
    return newcmp

def get_sweep360(heart_xy):    
    from scipy import ndimage
    def circular_sector(r_range, theta_range, LV_center):
        cx, cy = LV_center
        theta = theta_range/180*np.pi
        z = r_range.reshape(-1, 1).dot(np.exp(1.0j*theta).reshape(1, -1))
        xall = -np.imag(z) + cx
        yall = np.real(z) + cy
        return xall, yall
    
    LVwmask= (heart_xy==1)
    
    RVbmask = (heart_xy==2)
    LV_center = ndimage.center_of_mass(LVwmask)
    rr = np.min(np.abs(LVwmask.shape-np.array(LV_center))).astype(np.int)
    sweep360 = []
    for theta in range(360):
        #print(theta)
        xall, yall = circular_sector(np.arange(0, rr, 0.5),
                                     theta, LV_center)
        projection = ndimage.map_coordinates(RVbmask, [xall, yall], order=0).sum()
        sweep360.append(projection)
        
    
    return np.any(np.array(sweep360)==0)

def get_loc(num):
    mid = num//3
    basal = mid
    apical = mid
    if num % 3 == 1:
        mid = mid + 1
    elif num % 3 == 2:
        basal += 1
        mid += 1

    loc = [1]*basal + [2]*mid + [3]*apical
    
    return loc

def proc_slice_without_RV(mask1, mask5):
    #if np.unique(mask5).sum() == 6:
    #    seg = ahaseg.get_seg(mask5, nseg=6)
    #    return seg
    
    # find nearest point with ahaseg
    xx5, yy5 = np.nonzero(mask5==2)
    xx1, yy1 = np.nonzero(mask1)
    seg5 = mask5 * 0
    for ii in range(xx5.size):
        index = np.sqrt((xx1 - xx5[ii])**2 + (yy1 - yy5[ii])**2).argmin()
        seg5[xx5[ii], yy5[ii]] = mask1[xx1[index], yy1[index]]
    return seg5
def get_seg(heart_xyz):
    sum_slice = np.sum(heart_xyz==1, axis=(0, 1))
    temp = sum_slice.copy()
    temp = temp[temp>0]
    diff = temp[-1] - temp[0]

    sum_slice_all = np.sum(heart_xyz, axis=(0, 1))

    if diff < 0:
        basal_first = True
        #check apex
        apex_index = np.where(sum_slice_all>0)[0].max()    

    else:
        basal_first = False
        apex_index = np.where(sum_slice_all>0)[0].min()



    slice_name = np.zeros((heart_xyz.shape[2],))

    for slicen in range(heart_xyz.shape[2]):
        heart_xy = heart_xyz[..., slicen]
        #print(np.unique(heart_xy).sum())
        if np.sum(heart_xy, axis=(0, 1)) > 0:
            slice_name[slicen] = 11        
        if np.unique(heart_xy).sum() == 6 and (np.sum(heart_xy==3) > 20):           
            if get_sweep360(heart_xy):            
                slice_name[slicen] = 11
            else:
                slice_name[slicen] = 1
        LV = np.sum(heart_xy==1)
        LVall = LV + np.sum(heart_xy==2)
        LV_rate = LV/LVall
        if (apex_index == slicen) and (LV_rate < 0.05):
            slice_name[slicen] = 4
    slice_label = slice_name.astype(int).copy()
    temp_index = (slice_name==1) | (slice_name==11)
    slice_label[temp_index==True] = get_loc(np.sum(temp_index).astype(int))
    slice_true = slice_label.copy()
    slice_label[slice_name==11] = 11

    offset = dict()
    offset[1] = 0
    offset[2] = 6
    offset[3] = 12
    heart_aha6 = heart_xyz * 0
    heart_aha4 = heart_xyz * 0
    heart_aha = heart_xyz * 0
    slice_label_new = slice_label.copy()
    for ii in range(slice_label.size):
        if (slice_label[ii] == 0) or (slice_label[ii] == 11):
            continue

        if slice_label[ii] == 4:
            temp = heart_xyz[..., ii].copy()
            heart_aha[..., ii][temp==2] = 17
            continue

        seg6 = ahaseg.get_seg(heart_xyz[..., ii].copy(), 6)
        seg4 = ahaseg.get_seg(heart_xyz[..., ii].copy(), 4)

        #seg6[seg6 > 0] = seg6[seg6 > 0] + offset[slice_label[ii]]
        if np.sum(seg6) > 0:
            heart_aha6[..., ii] = seg6
            heart_aha4[..., ii] = seg4
        else:
            slice_label_new[ii] = 11

    while np.any(slice_label_new==11):
        #print(slice_label_new)
        for ii in np.where(slice_label_new==11)[0]:
            left = max(ii-1, 0)
            right = min(ii+1, slice_label_new.size-1)
            
            for test_index in [left, right]:
                #print(left, right)
                if slice_label_new[test_index] in [1, 2, 3]:
                    seg6 = proc_slice_without_RV(heart_aha6[..., test_index], heart_xyz[..., ii].copy())
                    seg4 = proc_slice_without_RV(heart_aha4[..., test_index], heart_xyz[..., ii].copy())
                    slice_label_new[ii] = slice_true[ii]
                    heart_aha6[..., ii] = seg6
                    heart_aha4[..., ii] = seg4
     
                
    for ii in range(slice_true.size):
        label = slice_true[ii]
        if label == 1:

            heart_aha[..., ii] = heart_aha6[..., ii]
        elif label == 2:
            seg = heart_aha6[..., ii]
            seg[seg > 0] = seg[seg > 0] + 6
            heart_aha[..., ii] = seg
        elif label == 3:
            seg = heart_aha4[..., ii]
            seg[seg > 0] = seg[seg > 0] + 12
            heart_aha[..., ii] = seg
    heart_aha[heart_aha > 0] = heart_aha[heart_aha > 0] + 4 
    heart_aha[heart_xyz==1] = 1
    heart_aha[heart_xyz==3] = 3
    return heart_aha


def convert(f, result_dir=None):
    if isinstance(f, str):
        temp = nibabel.load(f)
        heart_xyzt = temp.get_fdata()
        affine = temp.affine
        file_input = True
        new_f = basename(f)
    else:
        heart_xyzt = f.copy()
        file_input = False
        

    sys_frame, dia_frame = get_frame(heart_xyzt)


    heart_aha_4d = heart_xyzt * 0
    heart_aha_4d[..., sys_frame] = get_seg(heart_xyzt[..., sys_frame])
    heart_aha_4d[..., dia_frame] = get_seg(heart_xyzt[..., dia_frame])
    
    return heart_aha_4d

def safe_mkdir(dirname):
    try:
        os.mkdir(dirname)
    except:
        pass

def convert_mms(source_dir, dest_dir=None):

    cmap = get_cmap()
    
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
                montage(np.concatenate([heart_crop[..., sys_frame], heart_crop[..., dia_frame]], axis=-1)/21, 
                                        fname=jpg_file, display=False, cmap=cmap)
                img_file = f.replace('_gt.nii.gz', '.nii.gz')
                shutil.copyfile(img_file, join(result_dir, basename(img_file)))
                shutil.copyfile(f, join(result_dir, basename(f)))

            except Exception as e:

                print(e)
                print('error:', f)
                
                
                
def convert_acdc(source_dir, dest_dir=None):

    cmap = get_cmap()
    
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
        montage(np.concatenate([heart_crop[..., sys_frame], heart_crop[..., dia_frame]], axis=-1)/21,
                 fname=jpg_file, display=False, cmap=cmap)

        #except Exception as e:

        #    print(e)
        #    print('error:', f)
