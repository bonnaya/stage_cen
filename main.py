import numpy as np
from Traitement1imageRemovePlanPy import *
from Calcul_var import *
import matplotlib.ticker as ticker
import scipy.ndimage as nd


"""Ouverture du fichier"""

taille_police = 15
scale = 3472.59 / 1000  # résolution


def ouverture_fichier(file_name, scale):  # load data into numpy array
    data_init = np.genfromtxt(file_name, skip_header=14, usecols=[0, 1, 2])
    data = pd.DataFrame(data_init, columns=['x', 'y', 'z'])
    x = (data['x'] * scale).to_numpy().reshape(1000, 1000)
    y = (data['y'] * scale).to_numpy().reshape(1000, 1000)
    z = (data['z']).to_numpy().reshape(1000, 1000)
    return (x, y, z)


def traitement_img(x, y, z):
    z = remplissage(z)  # Remplacement des valeurs Nans de z
    z = retrait_pente2(x, y, z, ordre2)  # Retrait de la pente de l'échantillon, ordre à modifier
    z = rotation_img(z)  # Rotation de l'image pour avoir des sillons parallèles à la direction y
    z = crop_to_data(z)  # Elimination des Nans générés par la rotation de l'image
    z = outliers(z)
    z = initialisation(z)  # Mise à zéro du plan moyen de z
    (y, x) = np.mgrid[0:len(z), 0:len(z[0])]  # On remet x et y aux nouvelles dimensions de z.
    return x, y, z


def calculs_param(z, scale):
    Sa = rugosite_moyenne(z)  # Rugosité moyenne
    Sq = rugosite_moyenne_quadratique(z)  # Rugosité moyenne quadratique
    Ssk = skewness(z) / (Sq ** 3)  # Degré d'asymétrie du profil
    Sku = kurtosis(z) / (Sq ** 4)  # Kurtosis
    Ssm = mean_width(z) * scale  # Largeur moyenne des éléments de profil
    Sdq = rms_slope(z) / scale  # Pente moyenne
    (pente_x, pente_y) = np.divide(pentes_moy(z), scale)  # Pentes moyennes dans chque direction
    (Rk1, Rk2, pente_min, Mr1, Mr2, rpk, rvk) = rapport_matiere(z)  # Paramètres de hauteurs

    return Sa, Sq, Ssk, Sku, Ssm, Sdq, pente_x, pente_y, Rk1, Rk2, Mr1, Mr2, rpk, rvk, pente_min


""" Affichage des images """

""" Image en profilo """


def echelle(z):  #Mise en place d'une échelle commune aux échantillons de dimension similaire.
    p1 = int(np.percentile(z,
                           2))  # L'échelle est choisie en fonction de 98% des valeurs centrales (et non avec les valeurs extrêmes)
    p99 = int(np.percentile(z, 98))
    if max(-p1, p99) < 2.5:
        extremite = 3
        step = 0.1

    if 2.5 <= max(-p1, p99) < 5:
        extremite = 6
        step = 0.2

    if 5 <= max(-p1, p99) < 10:
        extremite = 12
        step = 0.5

    if 10 <= max(-p1, p99) < 15:
        extremite = 16
        step = 0.5

    if 15 <= max(-p1, p99) < 20:
        extremite = 21
        step = 1

    if 20 <= max(-p1, p99):
        extremite = 25
        step = 1


    return extremite, step


def img_profilo(x, y, z, file_name, taille_police, scale):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    cut_off = -0.5

    extremite, step = echelle(z)

    plt.title("Surface de l'echantillon")

    cntr = ax.contourf(x*scale , -y*scale , z, levels=np.arange(-extremite, extremite + step, step),
                           cmap='jet')  # Règle la répartition des couleurs assosicées aux valeurs

    cbar = fig.colorbar(cntr, ax=ax, shrink=1)
    cbar.set_label('z (µm)', size=taille_police)
    cbar.ax.tick_params(labelsize=0.75 * taille_police)  #police de la color bar
    ax.axis('on')
    ax.set_xlabel("x (µm)", fontsize=taille_police)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))  #échelle à changer
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_tick_params(which='major', labelsize=0.75 * taille_police, length=15, width=1)
    ax.xaxis.set_tick_params(which='minor', length=5, width=1)
    ax.set_ylabel("y (µm)", fontsize=taille_police)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1000))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_tick_params(which='major', labelsize=0.75 * taille_police, length=15, width=1)
    ax.yaxis.set_tick_params(which='minor', length=5, width=1)

    file_name = file_name.replace('.xyz', '')
    data_dir, filename = file_name.rsplit('/',
                                          1)  # On sépare le nom des dossiers de celui du fichier pour sauvegarder l'image dans le bon répertoire
    plt.savefig(data_dir + '/' + 'profilo/' + filename + '_profilo' + ".png", dpi=300)
    plt.close()



def courbe_distrib_hauteurs(z, file_name, taille_police):

    """ Affichage de la distribution des hauteurs"""

    fig2 = plt.figure(figsize=(8, 6))
    ax2 = plt.subplot(111)
    cut_off2 = -0.5


    (frequences, centres_bins) = distrib_hauteurs(z)  # Calcul de la distribution des hauteurs
    extremite, step = echelle(z)

    plt.title("Distribution des hauteurs")
    plt.plot(centres_bins, frequences, linestyle='-', color='black')    # Affichage de la distribution

    ax2.set_xlabel("z (µm)", fontsize=taille_police)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10 * step))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(step))
    ax2.xaxis.set_tick_params(which='major', labelsize=0.75 * taille_police, length=15, width=1)
    ax2.xaxis.set_tick_params(which='minor', length=5, width=1)
    ax2.set_ylabel("Probabilité", fontsize=taille_police)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax2.yaxis.set_tick_params(which='major', labelsize=0.75 * taille_police, length=15, width=1)
    ax2.yaxis.set_tick_params(which='minor', length=5, width=1)
    plt.grid()

    file_name = file_name.replace('.xyz', '')
    data_dir, filename = file_name.rsplit('/', 1)
    plt.savefig(data_dir + '/' + 'distrib/' + filename + '_distrib' + ".png", dpi=300)
    plt.close()



def courbe_rapport_matiere(z, Rk1, Rk2, Mr1, Mr2, rpk, rvk, pente_min, file_name, taille_police):
    """ Affichage de la courbe de rapport de matière (BAC curve)"""

    fig3 = plt.figure(figsize=(8, 6))
    ax3 = plt.subplot(111)
    cut_off3 = -0.5

    extremite, step = echelle(z)

    """Tracé d'une droite servant à déterminer Rk2 et Rk1, 
    c'est la sécante de plus faible pente avec 40% des hauteurs entre les deux intersections """

    z = z.flatten().tolist()  # Mise en forme de z sous forme de liste décroissante
    z.sort(reverse=True)
    proba = np.linspace(1 / len(z), 1,
                        len(z))  # Chaque altitude est reliée à une probabilite : ensemble de probas cumulées


    if len(pente_min) == 1 : # Répartition unimodale : une seule droite reliant Rk1 et Rk2
        x_droite = np.array([proba[0], proba[-1]])  # Tableau contenant les coordonnées de Rk2 et Rk2
        y_droite = np.array([Rk1, Rk2])
        plt.plot(x_droite, y_droite, color='red')  # Tracé de la sécante à partir de Rk1 et Rk2 appartenant à cette droite

    if len(pente_min) == 3 :
        pente_1, pente_2, i_moy = pente_min
        x_droite1 = np.array([proba[0], proba[i_moy]])  # Tableau contenant les coordonnées de Rk1 et Rk2
        y_droite1 = np.array([Rk1, pente_1 * proba[i_moy] + Rk1])
        plt.plot(x_droite1, y_droite1,
                 color='red')  # Tracé de la sécante à partir de Rk1 et Rk2 appartenant à cette droite

        x_droite2 = np.array([proba[i_moy], proba[-1]])  # Tableau contenant les coordonnées de Rk1 et Rk2
        y_droite2 = np.array([pente_2 * (proba[i_moy] - proba[-1]) + Rk2, Rk2])
        plt.plot(x_droite2, y_droite2,
                 color='red')  # Tracé de la sécante à partir de Rk1 et Rk2 appartenant à cette droite





    plt.title("Courbe de rapport de matière")
    plt.plot(0, Rk1, 1, Rk2, color="red", marker="+")
    plt.annotate(f'Rpk = {rpk:.2f}', xy=(0, Rk1), xytext=(0.02, Rk1 + 0.2 * step))
    plt.annotate(f'Rvk = {rvk:.2f}', xy=(1, Rk2), xytext=(0.8, Rk2 - 0.5*step))
    plt.annotate(f'Rk = {Rk1-Rk2:.2f}', xy=(0.5, 0), xytext=(0.5, 0))
    plt.vlines(x=Mr1, ymin=min(z), ymax=Rk1, linestyles='-', color='black')
    plt.annotate(f'Mr1 = {Mr1:.2f}', xy=(Mr1, 0), xytext=(Mr1 + 0.02, min(z)))
    plt.vlines(x=Mr2, ymin=min(z), ymax=Rk2, linestyles='-', color='black')
    plt.annotate(f'Mr2 = {Mr2:.2f}', xy=(Mr2, 0), xytext=(Mr2, min(z) - 0.2 * step))
    plt.hlines(xmin=0, xmax=Mr1, y=Rk1, linestyles='-', color='black')
    plt.hlines(xmin=Mr2, xmax=1, y=Rk2, linestyles='-', color='black')
    plt.fill_between(proba, z, Rk1, where=(proba <= Mr1), color='gray', alpha=0.5)
    plt.fill_between(proba, z, Rk2, where=(proba >= Mr2), color='gray', alpha=0.5)

    plt.plot(proba, z, linestyle='-', color='black')
    ax3.set_xlabel("Probabilités", fontsize=taille_police)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.2))  # échelle à changer
    ax3.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax3.xaxis.set_tick_params(which='major', labelsize=0.75 * taille_police, length=15, width=1)
    ax3.xaxis.set_tick_params(which='minor', length=5, width=1)
    ax3.set_ylabel("z (µm)", fontsize=taille_police)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(2 * step))
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(step))
    ax3.yaxis.set_tick_params(which='major', labelsize=0.75 * taille_police, length=15, width=1)
    ax3.yaxis.set_tick_params(which='minor', length=5, width=1)

    file_name = file_name.replace('.xyz', '')
    data_dir, filename = file_name.rsplit('/', 1)
    plt.savefig(data_dir + '/' + 'rapport_hauteur/' + filename + '_rapport_matiere' + ".png", dpi=300)
    plt.close()




