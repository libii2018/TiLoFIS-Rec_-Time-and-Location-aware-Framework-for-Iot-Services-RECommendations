#!/bin/bash

# Ce script exécute différentes versions de vos expériences KNN.
#
# Utilisation : ./run_experiments.sh
#
# Arguments pour main.py :
#   <numero_dataset> : 1 pour dataset1, 2 pour dataset2
#   <pourcentage> : Pourcentage des données à utiliser (ex: 100 pour 100%)
#   <type de donnees> : 1 temps de reponse , 2 debit
#   <type de modeles> : 1 base, 2 temps, 3 localisation, 4 temps + localisation


python main.py 2 0.01 1 1

python main.py 2 0.01 2 1

python main.py 2 0.01 1 2

python main.py 2 0.01 2 2

python main.py 2 0.01 1 3

python main.py 2 0.01 2 3

python main.py 2 0.01 1 4

python main.py 2 0.01 2 4



echo "Toutes les expériences sont terminées !"