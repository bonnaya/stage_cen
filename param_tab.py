import csv
from main import *
import os
from TraitementCartoPy import *
import seaborn as sns

"""Stockage des données dans un tableau csv et dans différents fichiers pour les images et graphes"""

""" Le programme fonctionne pour un dossier racine contenant des sous dossiers 
    (avec les catégories de semelle) et à l'intérieur des fichiers xyz"""

file_pattern = '*.xyz'
taille_police = 15
scale = 3472.59 / 1000  # résolution

i = int(input("Quelles données analyser ?\n 1 : figures\n 2 : Data_0406\n 3 : img_par_9 \n Saisir le numero : "))

if i == 1:
    data_dir = 'figures/'
    fichier_csv = 'parametres1.csv'

if i == 2:
    data_dir = 'Data_0406/'
    fichier_csv = 'parametres2.csv'


param = []

if i == 1 or i == 2:

    for data_dir1, data_dir2, files in os.walk(data_dir):
        for sous_dossier in data_dir2:
            dir_name = os.path.join(data_dir1, sous_dossier)
            for file in os.listdir(dir_name):
                if file.endswith('.xyz'):  # On vérifie que le fichier est un fichier xyz
                    filename = os.path.join(dir_name, file)
                    print(filename)
                    x, y, z = ouverture_fichier(filename, scale)
                    x, y, z = traitement_img(x, y, z)
                    Sa, Sq, Ssk, Sku, Ssm, Sdq, pente_x, pente_y, Rk1, Rk2, Mr1, Mr2, rpk, rvk, pente_min = calculs_param(z, scale)
                    hauteur = hauteur_moy_pic_sillon(z)
                    param.append([file] + [Sa, Sq, Ssk, Sku, Ssm, Sdq, pente_x, pente_y, Rk1 - Rk2, Mr1, Mr2, rpk, rvk, hauteur])

                    img_profilo(x, y, z, filename, taille_police, scale)
                    courbe_distrib_hauteurs(z, filename, taille_police)
                    courbe_rapport_matiere(z, Rk1, Rk2, Mr1, Mr2, rpk, rvk, pente_min, filename, taille_police)



if i == 3:
    data_dir = 'img_par_9/'
    fichier_csv = 'parametres3.csv'
    for data_dir1, data_dir2, files in os.walk(data_dir):
        for sous_dossier in data_dir2:
            if sous_dossier != 'distrib' and sous_dossier != 'rapport_hauteur' and sous_dossier != 'profilo':
                dir_name = os.path.join(data_dir1, sous_dossier)
                print(dir_name)
                (image_s, registration_couple_s, overlap_x, overlap_y) = open_file(dir_name + '/')
                image = registration_all(image_s, registration_couple_s, overlap_x, overlap_y)  # Collage des images

                z = crop_to_data(sitk.GetArrayFromImage(image))  # Coupe de l'image pour rendre les bords droits après l'alignement des images


                # Creation de tableaux x et y suivant le format des coordonnées des fichiers .xyz, aux dimensions de z
                x = np.tile(np.arange(len(z[0])), (len(z), 1))
                y = np.transpose(np.tile(np.arange(len(z)), (len(z[0]), 1)))
                x, y, z = traitement_img(x, y, z)


                Sa, Sq, Ssk, Sku, Ssm, Sdq, pente_x, pente_y, Rk1, Rk2, Mr1, Mr2, rpk, rvk, pente_min = calculs_param(z, scale)
                hauteur = hauteur_moy_pic_sillon(z)
                param.append([sous_dossier] + [Sa, Sq, Ssk, Sku, Ssm, Sdq, pente_x, pente_y, Rk1 - Rk2, Mr1, Mr2, rvk, rpk, hauteur])
                img_profilo(x, y, z, dir_name, taille_police, scale)
                courbe_distrib_hauteurs(z, dir_name, taille_police)
                courbe_rapport_matiere(z, Rk1, Rk2, Mr1, Mr2, rpk, rvk, pente_min, dir_name, taille_police)







"""Matrice de corrélation"""

param_struc = pd.DataFrame((row[-14:] for row in param) ,columns = ['Sa', 'Sq', 'Ssk', 'Sku', 'Ssm', 'Sdq', 'pente_x', 'pente_y', 'Rk', 'Mr1', 'Mr2', 'Rpk', 'Rvk', 'Pk_to_Vk'])
corr_moyenne = param_struc.corr('pearson')
corr_moyenne.round(2)

sns.heatmap(corr_moyenne, annot=True)
plt.savefig(data_dir +'_correlation' + ".png", dpi=300)
plt.show()
plt.close()








def conversion_csv(tab, fichier_csv):
    """
    Écrit un tableau dans un fichier CSV.
    """
    with open(fichier_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['', 'Sa', 'Sq', 'Ssk', 'Sku', 'Ssm', 'Sdq', 'pente_x', 'pente_y', 'Rk', 'Mr1', 'Mr2', 'Rpk', 'Rvk', 'Pk_to_Vk'])
        for ligne in tab:
            writer.writerow(ligne)

conversion_csv(param, data_dir + fichier_csv)



