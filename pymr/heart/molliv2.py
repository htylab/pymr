# -*- coding: utf-8 -*-

import torch
from scipy import ndimage
from os.path import join
import os
import numpy as np
import pydicom as dicom

def read_molli_dir(dirname):

    im=[]
    Invtime=[]
    count=0
    for fname in next(os.walk(dirname))[2]:
        fname_full = join(dirname,fname)
        try:
            temp = dicom.read_file(fname_full)
            im = np.append(im,temp.pixel_array)
            mat_m, mat_n = temp.pixel_array.shape
            Invtime = np.append(Invtime, temp.InversionTime)
            count += 1
        except:
            print("invalid dicom file, ignore this %s." % fname_full)
            pass

    #print("Total dicom file:%d" % count)
    im = np.reshape(im, (count, mat_m, mat_n))
    temp = np.argsort(Invtime)
    Invtime = Invtime[temp]
    im = im[temp]
    return im, Invtime



def synImg(synTI, dT1LL):

    Amap = dT1LL['Amap']
    Bmap = dT1LL['Bmap']
    T1starmap = dT1LL['T1starmap']

    j, k = Amap.shape
    Amap[T1starmap == 0] = 0
    Bmap[T1starmap == 0] = 0
    T1starmap[T1starmap == 0] = 1e-10

    if isinstance(synTI, np.ndarray):
        synImg = np.empty((synTI.size,j,k), dtype=np.double)
        for ii in range(synTI.size):
            synImg[ii] = abs(Amap - Bmap * np.exp(-synTI[ii] / T1starmap))
    else:
        synImg = Amap * 0.0
        synImg = abs(Amap - Bmap * np.exp(-synTI / T1starmap))

    return synImg

def T1LLmap(im, Invtime, threshold = 40):
    '''
    example call: T1map, Amap_pre, Bmap,T1starmap,Errmap = T1LLmap(im, Invtime)
    '''
    import ctypes
    import os
    import platform
    from numpy.ctypeslib import ndpointer
    import numpy as np

    TI_num = im.shape[0]
    mat_m = im.shape[1]
    mat_n = im.shape[2]
    mat_size = mat_m * mat_n



    if platform.system() == 'Windows':
        dllpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'T1map.dll')
        lb = ctypes.CDLL(dllpath)
        lib = ctypes.WinDLL(None, handle=lb._handle)
    else: #linux
        dllpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'T1map.so')
        lib = ctypes.CDLL(dllpath)

    syn = lib.syn    

    # Define the types of the output and arguments of this function.
    syn.restype = None
    syn.argtypes = [ndpointer(ctypes.c_double), 
                    ctypes.c_int,
                    ndpointer(ctypes.c_double),
                    ctypes.c_long,ctypes.c_double,
                    ndpointer(ctypes.c_double),
                    ndpointer(ctypes.c_double),
                    ndpointer(ctypes.c_double),
                    ndpointer(ctypes.c_double),
                    ndpointer(ctypes.c_double)]

    T1map = np.empty((1, mat_size), dtype=np.double)
    Amap = np.empty((1, mat_size), dtype=np.double)
    Bmap = np.empty((1, mat_size), dtype=np.double)
    T1starmap = np.empty((1, mat_size), dtype=np.double)
    Errmap = np.empty((1, mat_size), dtype=np.double)
    # We execute the C function, which will update the array.
    # 40 is threshold
    syn(Invtime, TI_num, im.flatten('f'), mat_size,
        threshold, T1map, Amap, Bmap, T1starmap, Errmap)


    if platform.system() == 'Windows':

        from ctypes.wintypes import HMODULE
        ctypes.windll.kernel32.FreeLibrary.argtypes = [HMODULE]
        ctypes.windll.kernel32.FreeLibrary(lb._handle)
    else:
        pass
        #lib.dlclose(lib._handle)

    dT1LL = dict()

    for key in ('T1map', 'Amap', 'Bmap', 'T1starmap', 'Errmap'):
        dT1LL[key] = np.reshape(locals()[key], (mat_m, mat_n), 'f')


    return dT1LL


def reg_spline_MI(fixed_np, moving_np, fixed_mask_np=None, meshsize=10):

    import SimpleITK as sitk

    fixed = sitk.Cast(sitk.GetImageFromArray(fixed_np), sitk.sitkFloat32)
    moving  = sitk.Cast(sitk.GetImageFromArray(moving_np), sitk.sitkFloat32)

    sitk.sitkUInt8
    transfromDomainMeshSize=[meshsize]*moving.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed,
                                          transfromDomainMeshSize )


    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(128)
    R.SetOptimizerAsGradientDescentLineSearch(learningRate=0.1,
                                              numberOfIterations=500,
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)
    R.SetOptimizerScalesFromPhysicalShift( )
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetShrinkFactorsPerLevel([6,2,1])
    R.SetSmoothingSigmasPerLevel([6,2,1])
    if not (fixed_mask_np is None):
        fixed_mask  = sitk.Cast(sitk.GetImageFromArray(fixed_mask_np),
                                sitk.sitkUInt8)
        R.SetMetricFixedMask(fixed_mask)

    #R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )
    #R.AddCommand( sitk.sitkMultiResolutionIterationEvent, lambda: command_multi_iteration(R) )

    outTx = R.Execute(fixed, moving)

    #print("-------")
    #print(outTx)
    #print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    #print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    #print(" Metric value: {0}".format(R.GetMetricValue()))
    #sitk.WriteTransform(outTx,  sys.argv[3])

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed);
    resampler.SetInterpolator(sitk.sitkLinear)
    #resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    moving_reg = resampler.Execute(moving)
    moving_reg_np = sitk.GetArrayFromImage(moving_reg)
    return moving_reg_np, resampler

def reg_move_it(resampler, img_np):
    import numpy as np
    import SimpleITK as sitk
    if len(img_np.shape) == 2:
        img  = sitk.Cast(sitk.GetImageFromArray(img_np), sitk.sitkFloat32)
        img_reg = resampler.Execute(img)
        img_reg_np = sitk.GetArrayFromImage(img_reg)
    else: #suppose to be 3
        img_reg_np=img_np.copy()
        for ii in range(img_np.shape[0]):
            img  = sitk.Cast(sitk.GetImageFromArray(img_np[ii]), sitk.sitkFloat32)
            img_reg = resampler.Execute(img)
            img_reg_np[ii] = sitk.GetArrayFromImage(img_reg)

    return img_reg_np


def rigid2d(fixed_np, moving_np, mask=None):
    import SimpleITK as sitk
    fixed = sitk.Cast(sitk.GetImageFromArray(fixed_np), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.GetImageFromArray(moving_np), sitk.sitkFloat32)

    R = sitk.ImageRegistrationMethod()

    if mask is not None:
        fixed_image_mask = sitk.Cast(
            sitk.GetImageFromArray(mask), sitk.sitkUInt8)
        R.SetMetricFixedMask(fixed_image_mask)
    R.SetMetricAsMattesMutualInformation(256)
    R.SetOptimizerAsGradientDescentLineSearch(learningRate=0.1,
                                              numberOfIterations=2000,
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)
    R.SetInitialTransform(sitk.Transform(
        fixed.GetDimension(), sitk.sitkTranslation))
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(fixed, moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    #resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out = resampler.Execute(moving)

    return sitk.GetArrayFromImage(out)


def reg2mean_rigid(im, mask=None, iters=4):

    try:
        meanim = np.mean(im, axis=0)
        if mask is None:
            mask = meanim * 0 + 1
        for jj in range(iters):
            regim = im*0

            for ii in range(regim.shape[0]):
                regim[ii] = rigid2d(meanim*mask, im[ii])

            meanim = np.mean(regim, axis=0)
    except:
        regim = im.copy()

    return regim


def reg2syn_spline(im, Invtime, iters=1):
    try:
        Img = im.copy()
        dT1LL = T1LLmap(Img, Invtime)
        for jj in range(iters):
            syn = synImg(Invtime, dT1LL)
            for ii in range(Img.shape[0]):
                Img[ii], _ = reg_spline_MI(
                    syn[ii], Img[ii], meshsize=7)
            dT1LL = T1LLmap(Img, Invtime)
    except:
        Img = im.copy()
    return Img


def crop(T1map, mask_rect):
    xx, yy = np.nonzero(mask_rect)

    return T1map[xx.min():xx.max(), yy.min():yy.max()]


def get_rect(mask, offset=5):
    xx, yy = np.nonzero(mask)

    xx_min = max(xx.min() - offset, 0)
    xx_max = min(xx.max() + offset, mask.shape[0])
    yy_min = max(yy.min() - offset, 0)
    yy_max = min(yy.max() + offset, mask.shape[1])

    mask_rect = mask * 0
    mask_rect[xx_min:xx_max, yy_min:yy_max] = 1

    return mask_rect





def T1seg(T1map_input, device='cpu'):

    def getLarea(input_mask):
        try:
            labeled_mask, cc_num = ndimage.label(input_mask)
            mask = (labeled_mask == (np.bincount(
                labeled_mask.flat)[1:].argmax() + 1))
        except:
            mask = input_mask
        return mask


    T1map = T1map_input.copy()
    T1map[T1map < 300] = 300
    T1map[T1map > 2500] = 2500

    NET = torch.jit.load("unet_T1300_2500_jit.pth", map_location=device)

    with torch.no_grad():
        image = T1map[None, ...][None, ...]
        image_d = torch.from_numpy(image).to(device).float()
        logits = NET(image_d).cpu().detach().numpy()
        mask_pred = np.argmax(logits[0, ...], axis=0)

    mask = (mask_pred * 0).astype(int)
    for ii in [1, 2, 3]:
        mask_temp = getLarea(mask_pred == ii)    
        mask[mask_temp > 0] = ii

    return mask


def AHAseg(mask, nseg=6):
    from scipy import ndimage

    def mid_to_angles(mid, seg_num):
        anglelist = np.zeros(seg_num)
        if seg_num == 4:
            anglelist[0] = mid - 45 - 90
            anglelist[1] = mid - 45
            anglelist[2] = mid + 45
            anglelist[3] = mid + 45 + 90

        if seg_num == 6:
            anglelist[0] = mid - 120
            anglelist[1] = mid - 60
            anglelist[2] = mid
            anglelist[3] = mid + 60
            anglelist[4] = mid + 120
            anglelist[5] = mid + 180
        anglelist = (anglelist + 360) % 360

        angles = np.append(anglelist, anglelist[0])
        angles = np.rad2deg(np.unwrap(np.deg2rad(angles)))

        return angles.astype(int)

    def circular_sector(theta_range, lvb):
        cx, cy = ndimage.center_of_mass(lvb)
        max_range = np.min(np.abs(lvb.shape-np.array([cx, cy]))).astype(np.int)
        r_range = np.arange(0, max_range, 0.1)
        theta = theta_range/180*np.pi
        z = r_range.reshape(-1, 1).dot(np.exp(1.0j*theta).reshape(1, -1))
        xall = -np.imag(z) + cx
        yall = np.real(z) + cy

        smask = lvb * 0
        xall = np.round(xall.flatten())
        yall = np.round(yall.flatten())
        mask = (xall >= 0) & (yall >= 0) & \
               (xall < lvb.shape[0]) & (yall < lvb.shape[1])
        xall = xall[np.nonzero(mask)].astype(int)
        yall = yall[np.nonzero(mask)].astype(int)
        smask[xall, yall] = 1

        return smask

    lvb = (mask == 1)
    lvw = (mask == 2)
    rvb = (mask == 3)
    lx, ly = ndimage.center_of_mass(lvb)
    rx, ry = ndimage.center_of_mass(rvb)
    j = (-1)**0.5
    lvc = lx + j*ly
    rvc = rx + j*ry
    mid_angle = np.angle(rvc - lvc)/np.pi*180 - 90

    angles = mid_to_angles(mid_angle, nseg)
    AHA_sector = lvw * 0
    for ii in range(angles.size-1):
        angle_range = np.arange(angles[ii], angles[ii + 1], 0.1)
        smask = circular_sector(angle_range, lvb)
        AHA_sector[smask > 0] = (ii + 1)

    label_mask = AHA_sector * lvw
    return label_mask

def T1LL_fit(TI0, y0):

    def T1LL(TI, *p):

        def T1LL_f(TI, A, B, T1_star):
            TI = float(TI)
            return ((A - (B * np.exp(-TI / T1_star))))

        A, B, T1_star = p
        if hasattr(TI, "__len__"):
            ally = TI*0.
            for ii, cTI in enumerate(TI):
                ally[ii] = T1LL_f(cTI, A, B, T1_star)
        else:
            ally = T1LL_f(TI, A, B, T1_star)
        return ally

    from scipy.optimize import curve_fit
    TI = TI0.copy()
    idx = np.argsort(TI)
    TI = TI[idx]

    err = 1e6
    T1 = 0
    params = 0
    for T1_guess in np.arange(800, 2200, 200):
        for kk in range(5):
            y = y0.copy()
            y = y[idx]
            y[:(kk+1)] = y[:(kk+1)]*-1
            try:
                params, cov = curve_fit(
                    T1LL, TI, y, p0=[y0[0], 2*y0[0], T1_guess])
                y1 = np.array([T1LL(ii, *params) for ii in TI])
                err1 = np.sum(np.abs(y1-y))
            except:
                err1 = 1e10

            if err1 < err:
                err = err1
                T1 = params[2]*(params[1]/params[0] - 1)
                params_final = params

    dT1LL = dict()
    dT1LL['A'] = params_final[0]
    dT1LL['B'] = params_final[1]
    dT1LL['T1*'] = params_final[2]
    dT1LL['T1'] = T1
    dT1LL['error'] = err

    return dT1LL


def fit_mask(im, invtime, mask, method='median'):
    T1_list = []
    idx_list = []
    error_list = []
    for idx in np.unique(mask):
        if idx == 0:
            continue  # ignore background 0
        curve = np.array([0] * im.shape[0])

        for jj in range(im.shape[0]):
            temp = im[jj][mask == idx]
            if method == 'mean':
                curve[jj] = np.mean(temp)
            else:
                curve[jj] = np.median(temp)

        dT1LL = T1LL_fit(invtime, curve)

        idx_list.append(idx)
        T1_list.append(dT1LL['T1'])
        error_list.append(dT1LL['error'])

    dT1 = dict()

    dT1['T1_list'] = T1_list
    dT1['idx_list'] = idx_list
    dT1['error_list'] = error_list
    dT1['method'] = method

    return dT1
