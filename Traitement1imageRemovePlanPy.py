# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 22:08:12 2022

"""

from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter
from Calcul_var import *
import SimpleITK as sitk
import scipy.ndimage as nd
from scipy.stats import zscore
import pandas as pd


def remplissage(z):

    """Remplacement de toutes les valeurs non définies du tableau"""

    for i in range(len(z)):
        for j in range(len(z[i])):
            k = 1
            while pd.isna(z[i][j]):
                if i - k + 1 <= 0:  # On traite les bords
                    z[i][j] = z[i + k][j]
                if i + k - 1 >= len(z[i]) - 1:
                    z[i][j] = z[i - k][j]  # Moyenne des plus proches voisins des colonnes (les sillons sont verticaux)
                else:
                    z[i][j] = (z[i - k][j] + z[i + k][j]) / 2
                k += 1
    return z


def initialisation(z):

    """On place le plan moyen à 0."""

    return z - np.nanmean(z)


def retrait_pente1(z):

    """Utilisation de la moyenne mobile, ok si sillons << taille échantillon"""

    moyenne_mobile = uniform_filter(z, 200, None, 'nearest')  # size à faire varier selon l'échantillon
    plt.imshow(moyenne_mobile)
    return z - moyenne_mobile


def ordre1(xy, a, b, c):

    """Equation d'une surface plane"""

    x, y = xy
    return a * x + b * y + c


def ordre2(xy, a, b, c, d, e, f):

    """Equation d'une surface quadratique"""

    x, y = xy
    return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + f

def ordre3 (xy, a, b, c, d, e, f, g, h, i ,j) :
    """ Equation d'une surface d'ordre 3"""

    x, y = xy
    return a * x ** 3 + b * y ** 3 + c * x**2 * y + d * x * y**2 + e * x **2 + f * y**2 + g * x * y + h * x + i * y + j


def ordre4(xy, a, b, c, d, e, f, g, h, i, j, k , l, m, n ,o):
    """
    :param xy:
    :param a:
    :param b:
    :param c:
    :param d:
    :param e:
    :param f:
    :param g:
    :param h:
    :param i:
    :param j:
    :param k:
    :param l:
    :param m:
    :param n:
    :param o:
    :return:
    Equation d'une surface d'ordre 3"""

    x, y = xy
    return a * x** 4 + b * y **4 + c * x**3 * y + d * x * y ** 3 + e * x**2 + y**2 + f * x ** 3 + g * y ** 3 + h * x ** 2 * y + i * x * y ** 2 + j * x ** 2 + k * y ** 2 + l * x * y + m * x + n * y + o

def retrait_pente2(x, y, z, f):

    """ Retrait d'une surface définie par une fonction"""

    x_list = x.flatten()  # Conversion des tableaux 2D sous forme de liste 1D
    y_list = y.flatten()
    z_list = z.flatten()

    popt, pcov = curve_fit(f, (x_list, y_list),
                           z_list)  # Recherche d'une surface minimisant la distance entre les points et la surface


    if len(popt) == 3:  # En fonction de l'ordre de la surface, le nombre de coefficients est différent
        a_opt, b_opt, c_opt = popt
        z_surf = ordre1((x, y), a_opt, b_opt, c_opt)  # Coefficients définissant la surface optimale

    if len(popt) == 6:
        a_opt, b_opt, c_opt, d_opt, e_opt, f_opt = popt
        z_surf = ordre2((x, y), a_opt, b_opt, c_opt, d_opt, e_opt, f_opt)

    if len(popt) == 10:
        a_opt, b_opt, c_opt, d_opt, e_opt, f_opt, g_opt, h_opt, i_opt, j_opt = popt
        z_surf = ordre3((x, y), a_opt, b_opt, c_opt, d_opt, e_opt, f_opt, g_opt, h_opt, i_opt, j_opt)

    if len(popt) == 15:
        a_opt, b_opt, c_opt, d_opt, e_opt, f_opt, g_opt, h_opt, i_opt, j_opt, k_opt, l_opt, m_opt, n_opt, o_opt = popt
        z_surf = ordre4((x, y), a_opt, b_opt, c_opt, d_opt, e_opt, f_opt, g_opt, h_opt, i_opt, j_opt, k_opt, l_opt, m_opt, n_opt, o_opt)

    """
    fig = plt.figure()                  # Afichage de la surface du ski et de la surface à retirer pour la mettre à plat
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z_surf, alpha=0.5, rstride=100, cstride=100, color='blue')
    ax.plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100, color='black')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    """
    return z - z_surf


def rotation_img(z):

    """calcul de l'angle pour lequel les variations de hauteur en y sont minimales
        Moins il y a de variations selon y et plus les sillons sont parallèles à l'axe y"""

    (pente_x, pente_min) = pentes_moy(z)  # Initialisation d'une variable qui stocke la pente minimum calculée

    z_lisse = nd.gaussian_filter(z, sigma=[25,2])  # Image lissée parallèlement aux sillon pour retirer le bruit
    z_im_lisse = sitk.GetImageFromArray(z_lisse)  # Conversion du tableau d'entrée en image
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(z_im_lisse.GetSpacing())
    resample.SetSize([len(z), len(z)])
    resample.SetOutputOrigin([0, 0])
    resample.SetDefaultPixelValue(np.nan)

    for i in range(100):  # Pente moyenne en y pour 200 angles allant de -5° à 5° par rapport à la position initiale.
        angle = 5 * i * np.pi / (100 * 180)

        resample.SetTransform(sitk.Euler2DTransform([500, 500], angle))  # Rotation dans le sens horaire
        z_pos = sitk.GetArrayFromImage(resample.Execute(z_im_lisse))

        resample.SetTransform(sitk.Euler2DTransform([500, 500], -angle))  # Rotation dans le sens trigo
        z_neg = sitk.GetArrayFromImage(resample.Execute(z_im_lisse))

        (p_x, pente_tmp_pos) = pentes_moy(z_pos)
        (p_x, pente_tmp_neg) = pentes_moy(z_neg)

        if min(pente_tmp_neg, pente_tmp_pos,
               pente_min) == pente_tmp_neg:  # z_min correspond au tableau pour lequel les sillons sont le plus droits
            pente_min = pente_tmp_neg
            angle_min = -angle

        if min(pente_tmp_neg, pente_tmp_pos, pente_min) == pente_tmp_pos:
            pente_min = pente_tmp_pos
            angle_min = angle

    z_im = sitk.GetImageFromArray(z)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(z_im.GetSpacing())  # Rotation de l'image non lissée
    resample.SetSize([len(z), len(z)])
    resample.SetOutputOrigin([0, 0])
    resample.SetDefaultPixelValue(np.nan)
    resample.SetTransform(sitk.Euler2DTransform([500, 500], angle_min))
    z_min = sitk.GetArrayFromImage(resample.Execute(z_im))

    return z_min


def crop_to_data(image):
    """
    Returns an image with the biggest rectangular region without NaN values.

    :param image:
    :return:
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

    return cropped_image


def outliers(z):

    """ Fonction qui retire les valeurs aberrantes
        On utilise un module qui se base sur une répartition Gaussienne des valeurs
        Les valeurs ne doivent pas se situer au delà d'un seuil (3 sigma dans ce cas)"""


    for i in range(0, len(z[0]), 20):  # Découpage de z en petits échantillons plus uniformes (variance moins élevée)
        if i + 20 < len(z[0]):  # Cas général quand ech est au centre du tableau
            ech = z[0:int(len(z)/2), i:i+20]   # Echantillon pris sur 10 colonnes et la moitié des lignes de z
            outlier = np.abs(zscore(ech)) > 3  # zscores : variable centrée réduite = (x-mu)/sigma
            ech[outlier] = np.nan
            z[0:int(len(z)/2), i:i+20] = ech


            ech = z[int(len(z)/2):len(z) - 1, i:i+20]  # Partie inférieure de z
            outlier = np.abs(zscore(ech)) > 3   # Si zscores > 4 écarts types, la valeur est considérée aberrante
            ech[outlier] = np.nan   # Remplacement des outliers par des Nans
            z[int(len(z)/2):len(z) - 1, i:i+20] = ech

        else:   # Dernière occuration : nb de colonnes après z[20*i] inférieur à 20
            ech = z[0:int(len(z) / 2), i:len(z)-1]
            outlier = np.abs(zscore(ech)) > 3
            ech[outlier] = np.nan
            z[0:int(len(z) / 2), i:len(z)-1] = ech

            ech = z[int(len(z) / 2):len(z) - 1, i:len(z)-1]
            outlier = np.abs(zscore(ech)) > 3
            ech[outlier] = np.nan
            z[int(len(z) / 2):len(z) - 1, i:len(z)-1] = ech

    z_rempli = remplissage(z)
    return z_rempli


