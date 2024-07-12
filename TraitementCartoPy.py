import numpy as np
import SimpleITK as sitk
import napari
from scipy.interpolate import NearestNDInterpolator
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from Traitement1imageRemovePlanPy import *

taille_police=15


def read_images(data_dir, im_prefix, im_suffix_s):
    """
    Reads a list of images and returns it in a 2D list.
    It also fills the nan value with nearest neighbor value.

    :param data_dir: folder where the images are stored
    :param im_prefix: prefix of the images
    :param im_suffix_s: 2D list describing the suffixes of the images
    :return:
    """

    image_s = []  # 2D list where to store the images

    for j in range(3):
        image_j = []
        for i in range(3):
            print(i, j)
            # Reading data
            file_name = data_dir + im_prefix + im_suffix_s[j][i]
            data = np.genfromtxt(file_name, skip_header=14, usecols=[2]).reshape(1000, 1000).astype('f4')

            # Filling nan values with nearest neighbor interpolator
            mask = np.where(~np.isnan(data))
            interp = NearestNDInterpolator(np.transpose(mask), data[mask])
            data_filled = interp(*np.indices(data.shape))

            # Translating into an ITK image with proper coordinates
            resolution = float(open(file_name, 'rt').readlines()[7].split(' ')[6]) * 1000  # mm
            image = sitk.GetImageFromArray(data_filled)
            image.SetSpacing([resolution, resolution])
            image.SetOrigin([i * 1000 * resolution, j * 1000 * resolution])
            image_j.append(image)

        image_s.append(image_j)

    return image_s



def crop_space(im_to_crop, im, margin):
    """
    Returns the region of im_to_crop that intersects the region of im (with extra margin if margin > 0)

    :param im_to_crop:
    :param im:
    :param margin:
    :return:
    """
    margin = np.array(margin)
    im_origin = np.array(im.GetOrigin())
    im_extent = im_origin + np.array(im.GetSize()) * np.array(im.GetSpacing())
    index_origin = np.array(im_to_crop.TransformPhysicalPointToIndex(im_origin - margin))
    index_origin = np.maximum(index_origin, np.array([0, 0]))
    index_extent = np.array(im_to_crop.TransformPhysicalPointToIndex(im_extent + margin))
    index_extent = np.minimum(index_extent, np.array(im_to_crop.GetSize()))

    size = np.array(index_extent - index_origin, dtype='int').tolist()
    index_origin = np.array(index_origin, dtype='int').tolist()

    im_cropped = sitk.RegionOfInterest(im_to_crop, size, index_origin)

    return im_cropped


def register_one(fixedImage, movingImage):
    """
    Registers movingImage to the reference fixedImage with 2D translations.
    Returns the registered image.

    :param fixedImage:
    :param movingImage:
    :return:
    """
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
    registration_method.SetMetricSamplingPercentage(1)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=1000,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    initial_transform = sitk.TranslationTransform(2, [0, 0, 0])
    registration_method.SetInitialTransform(initial_transform)

    registration_method.AddCommand(sitk.sitkIterationEvent,
                                   lambda: """print('Metric value:', registration_method.GetMetricValue(),
                                                 'Translation:', initial_transform.GetOffset())""")
    registration_method.Execute(fixedImage, movingImage)

    translation = initial_transform.GetOffset()
    movingImage.SetOrigin(np.array(movingImage.GetOrigin()) - np.array(translation))

    return movingImage


def viewer_image_2D_list(image_s):
    """
    Simple functions around napari to plot 2D list of images.

    :param image_s:
    :return:
    """
    viewer = napari.Viewer()

    for j in range(len(image_s)):
        for i in range(len(image_s[j])):
            im = image_s[j][i]
            args = {'scale': list(im.GetSpacing()[::-1]),
                    'translate': list(im.GetOrigin()[::-1]),
                    'contrast_limits': [85, 120]}
            viewer.add_image(sitk.GetArrayViewFromImage(im), **args)

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "mm"
    napari.run()


def registration_all(image_s, registration_couple_s, overlap_x, overlap_y):
    """
    Registration of 3x3 subimages into larger image.
    Tricky part may be to adjust overlap_x and y.

    :param image_s:
    :param registration_couple_s:
    :param overlap_x:
    :param overlap_y:
    :return:
    """
    for registration_couple in registration_couple_s:
        fixed_j, fixed_i = registration_couple[0]
        moving_j, moving_i = registration_couple[1]
        fixedImage = image_s[fixed_j][fixed_i]
        movingImage = image_s[moving_j][moving_i]
        resolution = fixedImage.GetSpacing()[0]
        translation = np.array([(moving_i - fixed_i) * (1000 - overlap_x), (moving_j - fixed_j) * (1000 - overlap_y)])
        movingImage.SetOrigin(np.array(fixedImage.GetOrigin()) + translation * resolution)
        movingImage = register_one(fixedImage, movingImage)

        intersect_moving = crop_space(movingImage, fixedImage, 0)
        intersect_fixed = crop_space(fixedImage, movingImage, 0)
        intersect_moving_avg = np.mean(sitk.GetArrayViewFromImage(intersect_moving))
        intersect_fixed_avg = np.mean(sitk.GetArrayViewFromImage(intersect_fixed))
        movingImage = movingImage + intersect_fixed_avg - intersect_moving_avg

        image_s[moving_j][moving_i] = movingImage

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(image_s[0][0].GetSpacing())
    resample.SetSize([3000, 3000])
    resample.SetOutputOrigin([-0.5, -0.5])
    resample.SetDefaultPixelValue(-1)

    image_array = np.zeros(shape=(3, 3, 3000, 3000))
    for j in range(len(image_s)):
        for i in range(len(image_s[j])):
            tab = sitk.GetArrayFromImage(resample.Execute(image_s[j][i]))
            tab[tab == -1] = np.nan
            image_array[j, i] = tab

    image_array = np.nanmean(image_array, axis=(0, 1))
    image = sitk.GetImageFromArray(image_array.astype('f4'))

    image.SetSpacing(image_s[0][0].GetSpacing())

    return image#_array


def filter_nan_gaussian_conserving(arr, sigma):
    """Apply a gaussian filter to an array with nans.

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = nd.gaussian_filter(loss, sigma=sigma, mode='constant', cval=1)

    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = nd.gaussian_filter(gauss, sigma=sigma, mode='constant', cval=0)
    gauss[nan_msk] = np.nan

    gauss += loss * arr

    return gauss


#def crop_to_data(image):
    """
    Returns an image with the biggest rectangular region without NaN values.

    :param image:
    :return:
    """
    """
    nans = np.isnan(image)  # Find position al all the NaNs
    nancols = np.all(nans, axis=0)  # Find all the columns that have only NaNs
    nanrows = np.all(nans, axis=1)  # Find all the columns that have only NaNs
    top_left_x = nancols.argmin()  # position of the left most column that does not contain all NaNs
    top_left_y = nanrows.argmin()  # position of the top most column that does not contain all NaNs
    cropped_image = image[:, ~nancols][~nanrows]  # remove all the rows and columns that are all NaNs

    while np.any(np.isnan(cropped_image)):  # Loop over the image until there a no NaNs left
        nans = np.isnan(cropped_image)  # Locate all NaNs
        nans_in_cols = np.sum(nans, axis=0)  # Figure out how many NaNs are in each column
        nans_in_rows = np.sum(nans, axis=1)  # Figure out how many NaNs are in each row
        if np.max(nans_in_cols) > np.max(nans_in_rows):
            # Remove the column or Row with the most NaNs, if it first row or column of the image,
            # add 1 to the top left x or y coordinate
            cropped_image = np.delete(cropped_image, np.argmax(nans_in_cols), 1)
            if np.argmax(nans_in_cols) == 0: top_left_x += 1
        else:
            cropped_image = np.delete(cropped_image, np.argmax(nans_in_rows), 0)
            if np.argmax(nans_in_rows) == 0: top_left_y += 1

    return cropped_image"""

#x = np.full((3200,3200),)




data_dir = 'figures/img_par_9/'

def open_file (data_dir):
    im_prefix = ''
    im_suffix_s = [['ImX0Y0.xyz', 'ImX1Y0.xyz', 'ImX2Y0.xyz'],
                   ['ImX0Y1.xyz', 'ImX1Y1.xyz', 'ImX2Y1.xyz'],
                   ['ImX0Y2.xyz', 'ImX1Y2.xyz', 'ImX2Y2.xyz']]
    image_s = read_images(data_dir, im_prefix, im_suffix_s)

    #viewer_image_2D_list(image_s)

    overlap_x, overlap_y = 100, 200
    registration_couple_s = [[[0, 0], [1, 0]],  # X0Y1 with X0YO
                             [[1, 0], [2, 0]],  # X0Y2 with X0Y1

                             [[0, 0], [0, 1]],  # X1Y0 with X0Y0
                             [[0, 1], [1, 1]],  # X1Y1 with X1Y0
                             [[1, 1], [2, 1]],  # X1Y2 with X1Y1

                             [[0, 1], [0, 2]],  # X2Y0 with X1Y0
                             [[0, 2], [1, 2]],  # X2Y1 with X2Y0
                             [[1, 2], [2, 2]],  # X2Y2 with X2Y1
                             ]
    return (image_s, registration_couple_s, overlap_x, overlap_y)



#image = registration_all(image_s, registration_couple_s, overlap_x, overlap_y)


#sitk.WriteImage(image, 'image.tif')
#image = sitk.ReadImage('image.tif')


"""

im = sitk.GetArrayFromImage(image)



#im = im - filter_nan_gaussian_conserving(im, sigma=(500, 500))  #Moyenne mobile ?
image_detrend = sitk.GetImageFromArray(im)





crp = crop_to_data(im)     # Tableau final contenant les images assemblees

x,y = np.where(crp>-1000)
print("x=",x)
print("y=",y)


image_cropped = sitk.GetImageFromArray(crp)
image_cropped.SetSpacing(image.GetSpacing())

fig=plt.figure(figsize=(8,6))
ax=plt.subplot(111)
#ax = axes([0,0,1,1], frameon=False)
ax.set_axis_off()
ax.set_xlim(0,10)
ax.set_ylim(0,10)
resolution=3.47259e-06*1000 #en mm
im = plt.imshow(crp, origin='upper',extent=[0,  im.shape[1] * resolution , 0,  im.shape[0] * resolution], cmap='jet')  # axes zoom in on portion of image
cbar = fig.colorbar(im,ax=ax,shrink=1)
plt.clim(-5, 10) 


cbar.set_label('z (µm)',size=taille_police)

cbar.ax.tick_params(labelsize=0.75*taille_police) #police de la color bar
ax.axis('on')
ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
ax.xaxis.tick_top()                     # and move the X-Axis
ax.set_title("x (mm)",fontsize=taille_police)
#ax.set_xlabel("x (µm)",fontsize=taille_police)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_tick_params(which='minor',length =5,width=1)
ax.xaxis.set_tick_params(which='major',labelsize = 0.75*taille_police,length =15,width=1)
ax.set_ylabel("y (mm)",fontsize=taille_police)
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_tick_params(which='major',labelsize = 0.75*taille_police,length =15,width=1)
ax.yaxis.set_tick_params(which='minor',length =5,width=1)
plt.savefig(data_dir+"/PyTot",dpi=300)

plt.show()

viewer_image_2D_list([[image, image_detrend, image_cropped]])
"""


