import csv
import pandas as pd
import numpy as np
import os




data_dir ='fart/'
fichier_csv = data_dir + 'proportion_fluor.csv'

prop_fluor = []


for data_dir1, data_dir2, files in os.walk(data_dir):
    for sous_dossier in data_dir2:
        if sous_dossier != 'Images':   # Images contient les fichiers qui ne sont pas utiles pour les calculs
            dir_name = os.path.join(data_dir1, sous_dossier)
            for file in os.listdir(dir_name):
                filename = os.path.join(dir_name, file)
                #print(filename)
                with open(filename, newline='') as csvfile:
                    data = pd.read_csv(csvfile, sep=',', header=3)
                    data = data.to_numpy()
                    data = data[:, 1:-1]  # Retrait des colonnes prises en trop (la première et la dernière)

                    if filename.endswith('-CPS.csv'):
                        CPS = data
                    if filename.endswith('-F.csv'):
                        fluor = data
            ratio = np.nanmean(fluor / CPS) # Proportion de fluor = la moyenne du ratio de fluor sur chaque pixel
            prop_fluor.append([sous_dossier] + [ratio])


def conversion_csv(tab, fichier_csv):
    """
    Écrit un tableau dans un fichier CSV.
    """
    with open(fichier_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Structure','Proportion de fluor sur la semelle'])
        for ligne in tab:
            writer.writerow(ligne)


conversion_csv(prop_fluor, fichier_csv) # Résultats stockés dans un fichier csv