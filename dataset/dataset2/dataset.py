import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm 

import os
import numpy as np
import pandas as pd
from collections import defaultdict 
from tqdm import tqdm 

class WSDREAMDataset:
    """
    Une classe pour charger et représenter le jeu de données QoS WS-DREAM (rtdata.txt et tpdata.txt).
    Suppose que les fichiers du jeu de données sont déjà disponibles localement.
    """
    def __init__(self, local_dataset_path, percentage: float = 1.0):
        """
        Initialise l'objet Dataset en spécifiant le chemin local où
        les fichiers du jeu de données (rtdata.txt, tpdata.txt) sont situés.

        Args:
            local_dataset_path (str): Le chemin du répertoire contenant
                                      les fichiers rtdata.txt et tpdata.txt.
                                      Exemple : 'wsdream_dataset2_data'
            percentage (float): Le pourcentage du jeu de données à utiliser, de 0.0 à 1.0.
                                La valeur par défaut est 1.0 (100 %).
        """
        if not os.path.isdir(local_dataset_path):
            raise FileNotFoundError(f"Le chemin spécifié n'existe pas ou n'est pas un répertoire : {local_dataset_path}")

        # Valider le pourcentage
        if not (0.0 <= percentage <= 1.0):
            raise ValueError("Le pourcentage doit être compris entre 0.0 et 1.0.")

        self.local_dataset_path = local_dataset_path
        self.percentage = percentage # Stocker le pourcentage pour une utilisation ultérieure
        self._rt_data_path = os.path.join(self.local_dataset_path, 'rtdata.txt')
        self._tp_data_path = os.path.join(self.local_dataset_path, 'tpdata.txt')

        # Dimensions du jeu de données selon le README
        self.num_users = 142
        self.num_services = 4500
        self.num_time_slices = 64

        self.rt_matrix_3d = None
        self.tp_matrix_3d = None
        self.rt_matrix_normalized = None
        self.tp_matrix_normalized = None
        self.rt_outlier_flags = None
        self.tp_outlier_flags = None
        
        # AJOUT : Initialisation de user_timestamps
        self.user_timestamps = defaultdict(dict) 

        print(f"Jeu de données initialisé. Recherche des fichiers dans : {self.local_dataset_path}")
        if self.percentage < 1.0:
            print(f"Seulement {self.percentage*100:.2f}% du jeu de données sera chargé.")

    def _normalize_min_max(self, data_matrix):
        """
        Applique la normalisation Min-Max à une matrice NumPy.
        Gère les valeurs de remplissage (ici, 0) en les ignorant pour le calcul min/max et en les préservant.
        """
        if data_matrix is None:
            return None

        print("Application de la normalisation Min-Max...")
        non_zero_values = data_matrix[data_matrix != 0]

        if len(non_zero_values) == 0:
            print("Avertissement : Aucune valeur non nulle trouvée pour la normalisation. La matrice normalisée sera entièrement à zéro.")
            return np.full_like(data_matrix, 0.0, dtype=float)

        min_val = np.min(non_zero_values)
        max_val = np.max(non_zero_values)

        if max_val == min_val:
            print("Avertissement : Les valeurs max et min sont identiques parmi les valeurs non nulles. La normalisation donnera des zéros pour les valeurs non nulles.")
            normalized_matrix = np.full_like(data_matrix, 0.0, dtype=float)
            return normalized_matrix
        else:
            normalized_matrix = np.zeros_like(data_matrix, dtype=float)
            non_zero_mask = (data_matrix != 0)
            normalized_matrix[non_zero_mask] = (data_matrix[non_zero_mask] - min_val) / (max_val - min_val)
            return normalized_matrix

    def _detect_and_assign_outliers(self, data_matrix):
        """
        Détecte les outliers en utilisant la méthode du boxplot et leur assigne une valeur binaire.
        - 1 si la valeur est supérieure au troisième quartile + 1.5*IQR.
        - 0 si la valeur est inférieure au premier quartile - 1.5*IQR.
        Les valeurs qui ne sont pas des outliers sont représentées par NaN dans la matrice de drapeaux.
        Gère les valeurs de remplissage (ici, 0) en les ignorant pour le calcul des quartiles.
        """
        if data_matrix is None:
            return None

        print("Détection et assignation des outliers en utilisant la méthode du boxplot...")
        flat_data = data_matrix[data_matrix != 0]

        if len(flat_data) < 2:
            print("Avertissement : Pas assez de points de données (hors zéros) pour détecter les outliers. Retourne des drapeaux NaN.")
            return np.full_like(data_matrix, np.nan)

        Q1 = np.percentile(flat_data, 25)
        Q3 = np.percentile(flat_data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_flags = np.full_like(data_matrix, np.nan, dtype=float)

        high_outlier_mask = (data_matrix > upper_bound) & (data_matrix != 0)
        outlier_flags[high_outlier_mask] = 1.0

        low_outlier_mask = (data_matrix < lower_bound) & (data_matrix != 0)
        outlier_flags[low_outlier_mask] = 0.0

        print(f"Détection des outliers terminée. {np.nansum(high_outlier_mask)} outliers hauts (valeur 1), {np.nansum(low_outlier_mask)} outliers bas (valeur 0).")
        return outlier_flags


    def load_rt_data_to_3d_matrix(self):
        """
        Charge un pourcentage spécifié du fichier rtdata.txt et le convertit en une matrice NumPy 3D.
        Les dimensions de la matrice sont (Tranche de temps, Utilisateur, Service).
        Les valeurs manquantes (éléments non sélectionnés par le pourcentage) sont représentées par 0.
        Ensuite, applique la normalisation min-max et détecte les outliers sur les valeurs non nulles.
        """
        if not os.path.exists(self._rt_data_path):
            raise FileNotFoundError(f"Fichier rtdata.txt non trouvé à l'adresse : {self._rt_data_path}")

        print(f"Lecture de {self._rt_data_path} et conversion en matrice 3D...")
        try:
            full_rt_df = pd.read_csv(self._rt_data_path, sep=r'\s+', header=None,
                                names=['ID Utilisateur', 'ID Service', 'ID Tranche de temps', 'Temps de réponse (sec)'])
            

            if self.percentage < 1.0:
                num_rows_to_use = int(len(full_rt_df) * self.percentage)
                if num_rows_to_use == 0 and len(full_rt_df) > 0:
                    num_rows_to_use = 1
                rt_df = full_rt_df.head(num_rows_to_use)
                print(f"Traitement des {len(rt_df)} premières lignes sur {len(full_rt_df)} lignes totales pour les données RT.")
            else:
                rt_df = full_rt_df
                print(f"Traitement de toutes les {len(rt_df)} lignes pour les données RT.")


            self.rt_matrix_3d = np.full((self.num_time_slices, self.num_users, self.num_services), 0.0, dtype=float)

            print("Remplissage de la matrice 3D (rt_matrix_3d) et de user_timestamps...")
            for index, row in tqdm(rt_df.iterrows(), total=len(rt_df), desc="Traitement des données RT"):
                user_idx = int(row['ID Utilisateur']) - 1
                service_idx = int(row['ID Service']) - 1
                time_slice_idx = int(row['ID Tranche de temps']) - 1
                response_time = row['Temps de réponse (sec)']

                if (0 <= time_slice_idx < self.num_time_slices and
                    0 <= user_idx < self.num_users and
                    0 <= service_idx < self.num_services):
                    self.rt_matrix_3d[time_slice_idx, user_idx, service_idx] = response_time
                    # AJOUT : Populer user_timestamps
                    self.user_timestamps[user_idx][service_idx] = time_slice_idx
            
            print(f"\nMatrice 3D 'rt_matrix_3d' créée avec la forme : {self.rt_matrix_3d.shape}")
            print(f"Nombre de zéros dans rt_matrix_3d (représentant les valeurs non échantillonnées) : {np.sum(self.rt_matrix_3d == 0)}")
            print(f"Nombre de valeurs non nulles dans rt_matrix_3d (représentant les valeurs échantillonnées) : {np.sum(self.rt_matrix_3d != 0)}")

            self.rt_matrix_normalized = self._normalize_min_max(self.rt_matrix_3d)
            if self.rt_matrix_normalized is not None:
                self.rt_outlier_flags = self._detect_and_assign_outliers(self.rt_matrix_normalized)
            else:
                self.rt_outlier_flags = None


        except pd.errors.EmptyDataError:
            print(f"Avertissement : Le fichier {self._rt_data_path} est vide.")
            self.rt_matrix_3d = np.full((self.num_time_slices, self.num_users, self.num_services), 0.0, dtype=float)
            self.rt_matrix_normalized = None
            self.rt_outlier_flags = None
        except Exception as e:
            print(f"Une erreur s'est produite lors de la lecture ou de la conversion de rtdata.txt : {e}")
            self.rt_matrix_3d = None
            self.rt_matrix_normalized = None
            self.rt_outlier_flags = None


    def load_tp_data_to_3d_matrix(self):
        """
        Charge un pourcentage spécifié du fichier tpdata.txt et le convertit en une matrice NumPy 3D.
        Les dimensions de la matrice sont (Tranche de temps, Utilisateur, Service).
        Les valeurs manquantes (éléments non sélectionnés par le pourcentage) sont représentées par 0.
        Ensuite, applique la normalisation min-max et détecte les outliers sur les valeurs non nulles.
        """
        if not os.path.exists(self._tp_data_path):
            print(f"Avertissement : Fichier tpdata.txt non trouvé à l'adresse : {self._tp_data_path}. Le chargement sera ignoré.")
            self.tp_matrix_3d = None
            self.tp_matrix_normalized = None
            self.tp_outlier_flags = None
            return

        print(f"Lecture de {self._tp_data_path} et conversion en matrice 3D...")
        try:
            full_tp_df = pd.read_csv(self._tp_data_path, sep=r'\s+', header=None,
                                names=['ID Utilisateur', 'ID Service', 'ID Tranche de temps', 'Débit (kbps)'])

            if self.percentage < 1.0:
                num_rows_to_use = int(len(full_tp_df) * self.percentage)
                if num_rows_to_use == 0 and len(full_tp_df) > 0:
                    num_rows_to_use = 1
                tp_df = full_tp_df.head(num_rows_to_use)
                print(f"Traitement des {len(tp_df)} premières lignes sur {len(full_tp_df)} lignes totales pour les données TP.")
            else:
                tp_df = full_tp_df
                print(f"Traitement de toutes les {len(tp_df)} lignes pour les données TP.")


            self.tp_matrix_3d = np.full((self.num_time_slices, self.num_users, self.num_services), 0.0, dtype=float)

            print("Remplissage de la matrice 3D (tp_matrix_3d)...")
            for index, row in tqdm(tp_df.iterrows(), total=len(tp_df), desc="Traitement des données TP"):
                user_idx = int(row['ID Utilisateur']) - 1
                service_idx = int(row['ID Service']) - 1
                time_slice_idx = int(row['ID Tranche de temps']) - 1
                throughput = row['Débit (kbps)']

                if (0 <= time_slice_idx < self.num_time_slices and
                    0 <= user_idx < self.num_users and
                    0 <= service_idx < self.num_services):
                    self.tp_matrix_3d[time_slice_idx, user_idx, service_idx] = throughput


            print(f"\nMatrice 3D 'tp_matrix_3d' créée avec la forme : {self.tp_matrix_3d.shape}")
            print(f"Nombre de zéros dans tp_matrix_3d (représentant les valeurs non échantillonnées) : {np.sum(self.tp_matrix_3d == 0)}")
            print(f"Nombre de valeurs non nulles dans tp_matrix_3d (représentant les valeurs échantillonnées) : {np.sum(self.tp_matrix_3d != 0)}")


            self.tp_matrix_normalized = self._normalize_min_max(self.tp_matrix_3d)
            if self.tp_matrix_normalized is not None:
                self.tp_outlier_flags = self._detect_and_assign_outliers(self.tp_matrix_normalized)
            else:
                self.tp_outlier_flags = None

        except pd.errors.EmptyDataError:
            print(f"Avertissement : Le fichier {self._tp_data_path} est vide.")
            self.tp_matrix_3d = np.full((self.num_time_slices, self.num_users, self.num_services), 0.0, dtype=float)
            self.tp_matrix_normalized = None
            self.tp_outlier_flags = None
        except Exception as e:
            print(f"Une erreur s'est produite lors de la lecture ou de la conversion de tpdata.txt : {e}")
            self.tp_matrix_3d = None
            self.tp_matrix_normalized = None
            self.tp_outlier_flags = None

# --- Exemple d'utilisation (similaire à avant, pour les tests) ---

# if __name__ == "__main__":
#     local_data_dir = 'wsdream_dataset2_data'

#     # Créer des fichiers factices pour les tests locaux s'ils n'existent pas
#     if not os.path.exists(local_data_dir):
#         print(f"Le dossier '{local_data_dir}' n'existe pas. Création pour l'exemple.")
#         os.makedirs(local_data_dir)
#         with open(os.path.join(local_data_dir, 'rtdata.txt'), 'w') as f:
#             f.write("98 4352 33 0.311\n")
#             f.write("91 1196 62 0.500\n")
#             f.write("1 1 1 0.1\n")
#             f.write("142 4500 64 1.234\n")
#             # Ajouter plus de lignes pour voir l'effet du pourcentage (par exemple, 10000 lignes)
#             for i in range(1, 10001):
#                 f.write(f"{i % 142 + 1} {i % 4500 + 1} {i % 64 + 1} {0.05 + i/10000.0}\n")
#             # Ajouter des outliers clairs pour RT
#             f.write("50 2000 30 10.0\n") # Outlier haut
#             f.write("10 100 5 0.0001\n") # Outlier bas
#             f.write("140 4400 60 15.0\n") # Un autre outlier haut


#         with open(os.path.join(local_data_dir, 'tpdata.txt'), 'w') as f:
#              f.write("91 1196 62 32.88\n")
#              f.write("1 1 1 100.0\n")
#              # Ajouter plus de lignes pour les données TP également
#              for i in range(1, 10001): # Taille des données factices augmentée pour TP également
#                  f.write(f"{i % 142 + 1} {i % 4500 + 1} {i % 64 + 1} {10.0 + i/100.0}\n")
#              # Ajouter des outliers clairs pour TP
#              f.write("50 2000 30 1.0\n") # Outlier bas
#              f.write("10 100 5 10000.0\n") # Outlier haut


#     print("Initialisation du jeu de données...")
#     try:
#         # Exemple 1 : Charger 100 % du jeu de données (comportement par défaut)
#         print("\n--- Chargement de 100% du jeu de données avec Normalisation et Outliers (valeurs manquantes -> 0) ---")
#         dataset_full = WSDREAMDataset(local_data_dir)
#         dataset_full.load_rt_data_to_3d_matrix()
#         dataset_full.load_tp_data_to_3d_matrix()

#         if dataset_full.rt_matrix_normalized is not None:
#             print("\nAccès à la matrice normalisée RT (rt_matrix_normalized) :")
#             print(f"Forme : {dataset_full.rt_matrix_normalized.shape}")
#             # Remarque : avec 0 comme valeur de remplissage, np.min et np.max incluront les 0.
#             # Utiliser data_matrix[data_matrix != 0] pour obtenir des statistiques sur les valeurs réelles.
#             # Vérifier s'il existe des valeurs non nulles avant d'appeler min/max
#             if np.sum(dataset_full.rt_matrix_normalized != 0) > 0:
#                 print(f"Min (non-nul) : {np.min(dataset_full.rt_matrix_normalized[dataset_full.rt_matrix_normalized != 0]):.4f}, Max (non-nul) : {np.max(dataset_full.rt_matrix_normalized[dataset_full.rt_matrix_normalized != 0]):.4f}")
#             else:
#                 print("Aucune valeur non nulle dans la matrice RT normalisée.")


#             print("\nExemple de valeurs normalisées et drapeaux d'outliers (quelques points) :")
#             # Essayons de trouver une valeur non nulle
#             non_zero_indices_rt = np.argwhere(dataset_full.rt_matrix_3d != 0)
#             if len(non_zero_indices_rt) > 0:
#                 sample_idx_rt = non_zero_indices_rt[0] # Prendre le premier index non-nul
#                 print(f"RT (Tranche={sample_idx_rt[0]+1}, Utilisateur={sample_idx_rt[1]+1}, Service={sample_idx_rt[2]+1}) :")
#                 print(f"  Valeur brute : {dataset_full.rt_matrix_3d[tuple(sample_idx_rt)]:.4f}")
#                 print(f"  Valeur normalisée : {dataset_full.rt_matrix_normalized[tuple(sample_idx_rt)]:.4f}")
#                 print(f"  Drapeau Outlier : {dataset_full.rt_outlier_flags[tuple(sample_idx_rt)]}")
            
#             # Vérifier une valeur zéro
#             zero_indices_rt = np.argwhere(dataset_full.rt_matrix_3d == 0)
#             if len(zero_indices_rt) > 0:
#                 zero_sample_idx_rt = zero_indices_rt[0]
#                 print(f"RT (Tranche={zero_sample_idx_rt[0]+1}, Utilisateur={zero_sample_idx_rt[1]+1}, Service={zero_sample_idx_rt[2]+1}) (Valeur zéro) :")
#                 print(f"  Valeur brute : {dataset_full.rt_matrix_3d[tuple(zero_sample_idx_rt)]:.4f}")
#                 print(f"  Valeur normalisée : {dataset_full.rt_matrix_normalized[tuple(zero_sample_idx_rt)]:.4f}")
#                 print(f"  Drapeau Outlier : {dataset_full.rt_outlier_flags[tuple(zero_sample_idx_rt)]} (Devrait être NaN car ce n'est pas une vraie mesure)")


#         if dataset_full.tp_matrix_normalized is not None:
#             print("\nAccès à la matrice normalisée TP (tp_matrix_normalized) :")
#             print(f"Forme : {dataset_full.tp_matrix_normalized.shape}")
#             # Vérifier s'il existe des valeurs non nulles avant d'appeler min/max
#             if np.sum(dataset_full.tp_matrix_normalized != 0) > 0:
#                 print(f"Min (non-nul) : {np.min(dataset_full.tp_matrix_normalized[dataset_full.tp_matrix_normalized != 0]):.4f}, Max (non-nul) : {np.max(dataset_full.tp_matrix_normalized[dataset_full.tp_matrix_normalized != 0]):.4f}")
#             else:
#                 print("Aucune valeur non nulle dans la matrice TP normalisée.")
            
#             non_zero_indices_tp = np.argwhere(dataset_full.tp_matrix_3d != 0)
#             if len(non_zero_indices_tp) > 0:
#                 sample_idx_tp = non_zero_indices_tp[0] # Prendre le premier index non-nul
#                 print(f"TP (Tranche={sample_idx_tp[0]+1}, Utilisateur={sample_idx_tp[1]+1}, Service={sample_idx_tp[2]+1}) :")
#                 print(f"  Valeur brute : {dataset_full.tp_matrix_3d[tuple(sample_idx_tp)]:.4f}")
#                 print(f"  Valeur normalisée : {dataset_full.tp_matrix_normalized[tuple(sample_idx_tp)]:.4f}")
#                 print(f"  Drapeau Outlier : {dataset_full.tp_outlier_flags[tuple(sample_idx_tp)]}")
            
#             # Vérifier une valeur zéro
#             zero_indices_tp = np.argwhere(dataset_full.tp_matrix_3d == 0)
#             if len(zero_indices_tp) > 0:
#                 zero_sample_idx