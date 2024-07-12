import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np







param = []






















"""
for i in range len(data): 
    param.append(data.iloc[:, i].values)
    ecarts_types = data.iloc[:, i+1].values

# Générer les labels des paramètres
parametres = [f'Param{i+1}' for i in range(len(valeurs))]

# Position des barres
x_pos = np.arange(len(parametres))

# Création de l'histogramme
plt.bar(x_pos, valeurs, yerr=ecarts_types, align='center', alpha=0.7, ecolor='black', capsize=10)

# Ajout des labels et du titre
plt.xlabel('Paramètres')
plt.ylabel('Valeurs')
plt.title('Histogramme des paramètres avec barres d\'erreur')

# Définir les labels des x-ticks
plt.xticks(x_pos, parametres)

# Affichage du graphique
plt.tight_layout()
plt.show()"""