
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import scipy.ndimage as nd

# Calcul de l'écart arithmétique à la valeur moyenne de la surface


def rugosite_moyenne(z) :
    return np.nanmean(np.abs(z))

def rugosite_moyenne_quadratique(z) :
    return np.sqrt(np.mean(np.square(z)))


def skewness(z) :
    return np.mean(np.power(z,3))

def kurtosis(z) :
    return np.mean(np.power(z,4))

def mean_width(z) :
    periods = []
    for i in range(0, len(z)):
        ind_period = np.where((z[i][:-1] >= 0) & (z[i][1:] < 0))[0]     #On cherche les endroits où les valeurs sont positives puis négatives : c'est une période
        periods.append(np.mean(np.diff(ind_period)))                    # Ce tableau stocke la largeur moyenne de chaque période sur une tranche
    return np.mean(periods)                                             # Largeur moyenne des périodes sur tot l'échantillon



def slopes(z):
    pente_x = np.diff(z)  # Pente en x = delta_z/delta_x = (z_i+1 - z_i) / 1 et diff(z) = z_i+1 - z_i
    z_inv = np.transpose(z)      # Pour y, on transpose la matrice pour avoir la pente
    pente_y = np.transpose(np.diff(z_inv)) # On remet la matrice à l'endroit
    return pente_x, pente_y


def rms_slope(z):
    (pente_x, pente_y) = slopes (z)
    return np.sqrt(np.divide((np.sum(np.square(pente_x))+np.sum(np.square(pente_y))), len(pente_x)*len(pente_x[0])+len(pente_y)*len(pente_y[0])))      #Pente moyenne totale


def pentes_moy(z) :                #Calcul de la pente moyenne en x et en y
    (pente_x, pente_y) = slopes(z)
    pente_moy_x = np.sqrt(np.nanmean(np.square(pente_x)))  #Moyenne quadratique
    pente_moy_y = np.sqrt(np.nanmean(np.square(pente_y)))
    #n = [[pente_moy_x**2, pente_moy_y*pente_moy_x], [pente_moy_x*pente_moy_y, pente_moy_y**2]]
    n = (pente_moy_x, pente_moy_y)
    return n



def distrib_hauteurs (z) :
    frequences, bins = np.histogram(z, bins=100, density=True)  # Calcul des fréquences correspondant à un intervalle de valeurs pour 100 intervalles
    centres_bins = (bins[:-1] + bins[1:]) / 2  # On associe un intervalle à un point
    return (frequences, centres_bins)



def rapport_matiere(z) :
    z = z.flatten().tolist()  # Mise en forme de z sous forme de liste décroissante
    z.sort(reverse=True)
    z = np.array(z)
    proba = np.linspace(1 / len(z), 1, len(z)) # Chaque valeur de z (de probabilité 1/len(z)) est associée à une case de proba qui augmente de 1/len(z) de case en case
    frequences, centres_bins = distrib_hauteurs(z)
    pk_to_vk = hauteur_med_pic_sillon(frequences, centres_bins)


    if pk_to_vk < 1: # Si les hauteurs max sont proches, la répartition est unimodale, on fait les calculs classiques
        pente_min = 1000000000000
        for i in range(int(len(z) - 0.4 * len(z))):
            pente_tmp = (z[i + int(0.4 * len(z))] - z[i]) / (proba[i + int(0.4 * len(z))] - proba[i])
            if abs(pente_tmp) < abs(pente_min):
                pente_min = pente_tmp
                ind_inf = i
                ind_sup = i + int(0.4 * len(z))

        Rk1 = z[ind_sup] - pente_min * proba[ind_sup]  # Rk1 est l'ordonnée à l'origine de la droite
        Rk2 = pente_min * proba[-1] + Rk1
        pente_min = [pente_min]

    else:   # Si la répartition est bimodale, on divise la courbe en deux parties pour avoir Rk1 et Rk2 dans chacune

        i_moy, = np.where((z[:-1] >= 0.0) & (z[1:] < 0.0))[0]
        z1 = z[0:i_moy]  # z1 permet de calculer Rk1
        z2 = z[i_moy:len(z) - 1]  # z2 permet de calculer Rk2
        pente_min1 = 1000000000000
        pente_min2 = 1000000000000

        for i in range(int(len(z1) - 0.4 * len(z1))):   # Calcul de Rk1 pour la 1ère distribution
            pente_tmp1 = (z1[i + int(0.4 * len(z1))] - z1[i]) / (proba[i + int(0.4 * len(z1))] - proba[i])
            if abs(pente_tmp1) < abs(pente_min1):
                pente_min1 = pente_tmp1
                i_min1 = i
        Rk1 = z1[i_min1] - pente_min1 * proba[i_min1]  # Rk1 est l'ordonnée à l'origine de la droite, lim sup de Rk

        for i in range(int(len(z2) - 0.4 * len(z2))):   # Calcul de Rk1 pour la 2ème distribution
            pente_tmp2 = (z2[i + int(0.4 * len(z2))] - z2[i]) / (proba[i + int(0.4 * len(z2))] - proba[i])
            if abs(pente_tmp2) < abs(pente_min2):
                pente_min2 = pente_tmp2
                i_min2 = i
        Rk2 = pente_min2 * (proba[-1] - proba[i_min2+len(z1)]) + z2[i_min2] #Rk2 est la lim inf de Rk
        pente_min = (pente_min1, pente_min2, i_moy)

    Mr1 = proba[np.argmax(z < Rk1)]     # Mr1 et Mr2 représentent la proportion de pics et la proportion de coeur
    Mr2 = proba[np.argmax(z < Rk2)]

    aire1 = 0
    aire_rpk = (z[0]-Rk1)*Mr1 /2
    i_rpk = 0
    aire2 = 0
    i_rvk = len(z)-1
    aire_rvk = Rk2 * (1-Mr2) / 2

    for i in range (int(Mr1 * len(z))) :
        aire1 += (z[i] * z[i+1])/(2*len(z)) # Calcul de l'aire pour Rvk en utilisant la méthode des trapèzes
    aire1 -= Mr1*Rk1

    while aire_rpk > aire1 :
        aire_rpk = z[i_rpk]*Rk1*Mr1 /2 - Rk1*Mr1
        i_rpk += 1
    rpk = z[i_rpk] - Rk1

    for i in range (int(Mr2 * len(z)), len(z)-1) :
        aire2 += ((Rk2 - z[i])* (Rk2 - z[i+1]))/(2*len(z)) # Calcul de l'aire pour Rvk en utilisant la méthode des trapèzes

    while aire_rvk > aire2 :
        aire_rvk = (Rk2-z[i_rvk])*(1-Mr2) /2
        i_rvk -= 1

    rvk = Rk2 - z[i_rvk]


    return Rk1, Rk2, pente_min, Mr1, Mr2, rpk, rvk


def aire_contact(z,h, scale) : #donne le pourcentage de l'aire de contact avec la neige en fonction de la pfondeur d'enfoncement
    aire_de_contact = np.sum(z>h)/(len(z[0])*len(z))              #Chaque élément>h représente un carré d'une unité d'aire

    print("z_contact = ", aire_de_contact*100, "%")


def hauteur_med_pic_sillon(frequences, centres_bins):

    """Hauteur pic-sillon avec la courbe de distribution des hauteurs
    Sert à savoir si une structure est lisse (rpk<1) ou linéaire (rpk>1)"""

    f_lisse = nd.gaussian_filter(frequences, sigma=3)
    i_moy, = np.where((centres_bins[:-1] < 0) & (centres_bins[1:] >= 0))[0]
    f_lisse1 = f_lisse[0:i_moy]  # Découpage en 2 fcts contenant chacune une hauteur max pour les fcts bi-modales
    f_lisse2 = f_lisse[i_moy:len(f_lisse) - 1]
    pk_to_vk = centres_bins[np.argmax(f_lisse2) + len(f_lisse1)] - centres_bins[np.argmax(f_lisse1)]

    return pk_to_vk

def hauteur_moy_pic_sillon(z) :
    """Hauteur moyenne du pic au sillon"""

    z_lisse = nd.gaussian_filter(z, sigma=[25,2]) # lissage de l'image afin de ne pas avoir la microrugosité, mais juste les pics et les sillons
    h_pic_sillon =[]
    ind_period = []
    for i in range (len(z)):
        ind_period = np.where((z_lisse[i][:-1] >= 0) & (z_lisse[i][1:] < 0))[0] # Indice séparant chaque période pour une ligne de a
        for j in range (len(ind_period)-1):
            z_period =z_lisse[i,int(ind_period[j]) : int(ind_period[j+1])]  # Recherche du max (pic) et du min (sillon) au sein d'une période
            h_pic_sillon.append(np.max(z_period)-np.min(z_period))

    h_moy = np.mean (h_pic_sillon)
    return h_moy