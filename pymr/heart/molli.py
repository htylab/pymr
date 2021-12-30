# -*- coding: utf-8 -*-

import os

import numpy as np
import pydicom as dicom


def read_molli_dir(dirname):

    im=[]
    Invtime=[]
    count=0
    for fname in next(os.walk(dirname))[2]:
        fname_full = os.path.join(dirname,fname)
        try:
            temp = dicom.read_file(fname_full)
            im = np.append(im,temp.pixel_array)
            mat_m, mat_n = temp.pixel_array.shape
            Invtime = np.append(Invtime, temp.InversionTime)
            count += 1
        except:
            print("invalid dicom file, ignore this %s." % fname_full)
            pass

    print("Total dicom file:%d" % count)
    im = np.reshape(im, (count, mat_m, mat_n))
    temp = np.argsort(Invtime)
    Invtime = Invtime[temp]
    im = im[temp]
    return im, Invtime



def synImg_cal2(synTI, Amap,Bmap, T1starmap):
    '''
    synTI = np.arange(0,2500,10)
    synImg = synImg_cal(synTI, Amap_pre, Bmap_pre, T1starmap_pre)
    '''
    j,k = Amap.shape
    Amap[np.nonzero(T1starmap==0)]=0
    Bmap[np.nonzero(T1starmap==0)]=0
    T1starmap[np.nonzero(T1starmap==0)]=1e-10

    if isinstance(synTI, np.ndarray):
        synImg = np.empty((synTI.size,j,k), dtype=np.double)
        for ii in range(synTI.size):
            synImg[ii] = abs(Amap-Bmap*np.exp(-synTI[ii]/T1starmap))
    else:
        synImg = Amap * 0.0
        synImg = abs(Amap-Bmap*np.exp(-synTI/T1starmap))


    return synImg

def T1LLmap(im, Invtime, threshold = 40):
    '''
    example call: T1map, Amap_pre, Bmap,T1starmap,Errmap = T1LLmap(im, Invtime)
    '''
    TI_num = im.shape[0]
    mat_m = im.shape[1]
    mat_n = im.shape[2]
    mat_size = mat_m * mat_n

    import ctypes
    import os
    import platform

    import numpy as np

    if platform.system() == 'Windows':
        dllpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'T1map.dll')
        lb = ctypes.CDLL(dllpath)
        lib = ctypes.WinDLL(None, handle=lb._handle)
    else: #linux
        dllpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'T1map.so')
        lib = ctypes.CDLL(dllpath)

    syn = lib.syn

    from numpy.ctypeslib import ndpointer

    # Define the types of the output and arguments of this function.
    syn.restype = None
    syn.argtypes = [ndpointer(ctypes.c_double),ctypes.c_int,
                  ndpointer(ctypes.c_double),ctypes.c_long,ctypes.c_double,
                  ndpointer(ctypes.c_double),ndpointer(ctypes.c_double),
                  ndpointer(ctypes.c_double),ndpointer(ctypes.c_double),ndpointer(ctypes.c_double)]

    T1map = np.empty((1,mat_size), dtype=np.double)
    Amap = np.empty((1,mat_size), dtype=np.double)
    Bmap = np.empty((1,mat_size), dtype=np.double)
    T1starmap = np.empty((1,mat_size), dtype=np.double)
    Errmap = np.empty((1,mat_size), dtype=np.double)
    # We execute the C function, which will update the array.
    # 40 is threshold
    syn(Invtime,TI_num, im.flatten('f'),mat_size, threshold, T1map, Amap, Bmap, T1starmap,Errmap)





    if platform.system() == 'Windows':

        from ctypes.wintypes import HMODULE
        ctypes.windll.kernel32.FreeLibrary.argtypes = [HMODULE]
        ctypes.windll.kernel32.FreeLibrary(lb._handle)
    else:
        pass
        #lib.dlclose(lib._handle)




    T1map=np.reshape(T1map,(mat_m,mat_n),'f')
    Amap=np.reshape(Amap,(mat_m,mat_n),'f')
    Bmap=np.reshape(Bmap,(mat_m,mat_n),'f')
    T1starmap=np.reshape(T1starmap,(mat_m,mat_n),'f')
    Errmap=np.reshape(Errmap,(mat_m,mat_n),'f')
    return T1map, Amap, Bmap,T1starmap,Errmap


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


def bloodmask_preGd(T1map_pre, lvsize_th = 300, display = False):
    bmask = np.bitwise_and(T1map_pre > 1500,T1map_pre < 3000)

    import skimage.measure as skim
    from scipy import ndimage




    label_img = skim.label(bmask)
    regions = skim.regionprops(label_img)

    ratio_max = 0
    count = 0
    for props in regions:

        if props.area > lvsize_th :
            count = count + 1
            x0,y0 = props.centroid
            dist = (((x0-(label_img.shape[0]/2))**2) + ((y0-(label_img.shape[1]/2))**2))**0.5
            ratio_temp = props.extent/dist*100
            #print((props.label,props.extent,props.area,props.filled_area,dist, ratio_temp))

            if ratio_temp > ratio_max:
                lv_selected_region = props.label
                #filled_image = props.filled_image
                x_lv,y_lv = props.centroid
                ratio_max = ratio_temp


    lvb_mask_rough = label_img == lv_selected_region

    ratio_max = 0
    count = 0
    for props in regions:

        if props.area > lvsize_th and (props.label != lv_selected_region):

            count = count +1
            #plt.figure(count)
            #plt.imshow(props.image)
            #fig, ax = plt.subplots()
            #ax.imshow(bmask, cmap=plt.cm.gray)
            #minr, minc, maxr, maxc = props.bbox
            #bx = (minc, maxc, maxc, minc, minc)
            #by = (minr, minr, maxr, maxr, minr)
            #ax.plot(bx, by, '-b', linewidth=2.5)

            x0,y0 = props.centroid
            dist = (((x0-x_lv)**2) + ((y0-y_lv)**2))**0.5
            ratio_temp = props.extent/dist*100
            #print((props.label != selected_region,props.label,props.extent,props.area,props.filled_area,dist, ratio_temp))


            if ratio_temp > ratio_max:
                rv_selected_region = props.label
                #filled_image = props.filled_image
                x_rv,y_rv = props.centroid
                ratio_max = ratio_temp


    rvb_mask_rough = label_img == rv_selected_region


    lvb_mask_rough= ndimage.binary_fill_holes(lvb_mask_rough.astype(int))
    lvb_mask_rough= ndimage.binary_closing(lvb_mask_rough.astype(int), structure=np.ones((5,5)))
    rvb_mask_rough= ndimage.binary_fill_holes(rvb_mask_rough.astype(int))
    rvb_mask_rough= ndimage.binary_closing(rvb_mask_rough.astype(int), structure=np.ones((5,5)))

    bmask_dict = {"lvb_mask":lvb_mask_rough, "lvb_centroid": (x_lv,y_lv),
                  "rvb_mask":rvb_mask_rough, "rvb_centroid": (x_rv,y_rv)}
    if display:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(18,10), dpi=300)
        plt.subplot(221),plt.imshow(lvb_mask_rough)
        plt.subplot(222),plt.imshow(rvb_mask_rough)



    return bmask_dict
