import matplotlib.pyplot as plt
import numpy as np


def montage(X, cmap='viridis', fname=None, display=True, interpolation='nearest', vmin=None, vmax=None):
    m, n, count = X.shape    
    mm = int(np.ceil(np.sqrt(count)))
    nn = mm
    M = np.zeros((mm * m, nn * n))

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count: 
                break
            sliceM, sliceN = j * m, k * n
            M[sliceM:sliceM + m, sliceN:sliceN + n] = X[:, :, image_id]
            image_id += 1
                    
    #pylab.imshow(flipud(rot90(M)), cmap=colormap)
    #pylab.axis('off')
    if display == False:
        plt.ioff()
    plt.figure(figsize=(10, 10))
    plt.imshow(M, cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax)
    plt.axis('off')
    if fname is not None:
        plt.savefig(fname)
    return M



def plot_aha17(aha17, vmin=None, vmax=None, fname=None):
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if vmin is None:
        vmin = 0
        vmax = np.max(aha17)
    
        #Create the fake data
    #data = np.array(range(17))*10 + 1


    # Make a figure and axes with dimensions as desired.
    fig=plt.figure(figsize=(9,8), dpi=200)

    #fig, ax = plt.subplots(figsize=(8,12),nrows=1, ncols=1,
    #                       subplot_kw=dict(projection='polar'))
    #fig.canvas.set_window_title('Left Ventricle Bulls Eyes (AHA)')

    # Create the axis for the colorbars
    ax = fig.add_axes([0.1, 1-8/9.+0.1, 8./9.*0.7, 1*0.7],projection='polar')
    axl = fig.add_axes([0.76, 0.3, 0.03, 0.5])
    #axl = fig.add_axes([0.76, 0.3, 0.01, 0.5])
    #axl.set_aspect(1)
    #axl2 = fig.add_axes([0.41, 0.15, 0.2, 0.05])
    #axl3 = fig.add_axes([0.69, 0.15, 0.2, 0.05])


    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.

    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    #norm = mpl.colors.Normalize(vmin=0, vmax=256)
    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm,
                                    orientation='vertical')
    #cb1.set_label('A. U.')


    # Create the 17 segment model
    bullseye_plot(ax, aha17, cmap=cmap, norm=norm)

    #ax.set_title('Bulls Eye (AHA)')

    #plt.clf()

    plt.show()
    if fname is not None:
        plt.savefig(fname)


def bullseye_plot(ax, data, segBold=[], cmap=None, norm=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    """
    Bullseye representation for the left ventricle.

    Parameters
    ------------
    ax : axes
    data : list of int and float
        The intensity values for each of the 17 segments
    cmap : ColorMap or None
        Optional argument to set the disaried colormap
    norm : Normalize or None
        Optional argument to normalize data into the [0.0, 1.0] range
    segBold: list of int
        A list with the segments to highlight


    Notes
    -----
    This function create the 17 segment model for the left ventricle according
    to the American Heart Association (AHA) [1]_

    References
    ----------
    .. [1] M. D. Cerqueira, N. J. Weissman, V. Dilsizian, A. K. Jacobs,
        S. Kaul, W. K. Laskey, D. J. Pennell, J. A. Rumberger, T. Ryan,
        and M. S. Verani, "Standardized myocardial segmentation and nomenclature
        for tomographic imaging of the heart", Circulation, vol. 105, no. 4,
        pp. 539-542, 2002.
    """

    linewidth = 2
    data = np.array(data).ravel()

    if cmap is None:
        cmap = plt.cm.jet

    if norm is None:
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    theta = np.linspace(0, 2*np.pi, 768)
    r = np.linspace(0.2, 1, 4)

    # Create the bound for the segment 17
    for i in range(r.shape[0]):
        ax.plot(theta, np.repeat(r[i], theta.shape), '-k', lw=linewidth)

    # Create the bounds for the segments  1-12
    for i in range(6):
        theta_i = i*60*np.pi/180
        ax.plot([theta_i, theta_i], [r[1], 1], '-k', lw=linewidth)

    # Create the bounds for the segmentss 13-16
    for i in range(4):
        theta_i = i*90*np.pi/180 - 45*np.pi/180
        ax.plot([theta_i, theta_i], [r[0], r[1]], '-k', lw=linewidth)

    # Fill the segments 1-6
    r0 = r[2:4]
    r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T
    for i in range(6):
        # First segment start at 60 degrees
        theta0 = theta[i*128:i*128+128] + 60*np.pi/180
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((128, 2))*data[i]
        ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm)
        if i+1 in segBold:
            ax.plot(theta0, r0, '-k', lw=linewidth+2)
            ax.plot(theta0[0], [r[2], r[3]], '-k', lw=linewidth+1)
            ax.plot(theta0[-1], [r[2], r[3]], '-k', lw=linewidth+1)

    # Fill the segments 7-12
    r0 = r[1:3]
    r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T
    for i in range(6):
        # First segment start at 60 degrees
        theta0 = theta[i*128:i*128+128] + 60*np.pi/180
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((128, 2))*data[i+6]
        ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm)
        if i+7 in segBold:
            ax.plot(theta0, r0, '-k', lw=linewidth+2)
            ax.plot(theta0[0], [r[1], r[2]], '-k', lw=linewidth+1)
            ax.plot(theta0[-1], [r[1], r[2]], '-k', lw=linewidth+1)

    # Fill the segments 13-16
    r0 = r[0:2]
    r0 = np.repeat(r0[:, np.newaxis], 192, axis=1).T
    for i in range(4):
        # First segment start at 45 degrees
        theta0 = theta[i*192:i*192+192] + 45*np.pi/180
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((192, 2))*data[i+12]
        ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm)
        if i+13 in segBold:
            ax.plot(theta0, r0, '-k', lw=linewidth+2)
            ax.plot(theta0[0], [r[0], r[1]], '-k', lw=linewidth+1)
            ax.plot(theta0[-1], [r[0], r[1]], '-k', lw=linewidth+1)

    #Fill the segments 17
    if data.size == 17:
        r0 = np.array([0, r[0]])
        r0 = np.repeat(r0[:, np.newaxis], theta.size, axis=1).T
        theta0 = np.repeat(theta[:, np.newaxis], 2, axis=1)
        z = np.ones((theta.size, 2))*data[16]
        ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm)
        if 17 in segBold:
            ax.plot(theta0, r0, '-k', lw=linewidth+2)

    ax.set_ylim([0, 1])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
