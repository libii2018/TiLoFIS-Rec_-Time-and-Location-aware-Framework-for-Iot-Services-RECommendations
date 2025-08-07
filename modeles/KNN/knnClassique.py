import numpy as np
import pandas as pd
import math
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import defaultdict
from sklearn.cluster import KMeans, AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm 
from collections import defaultdict


# --- Classe KNNClassique ---

class KNNClassique:
    def __init__(self, k, similarity_metric='pearson', approach='user_based', 
                 normalize_ratings=True, 
                 apply_location_similarity=False, # Conservez ce paramètre, mais son effet dépendra des données
                 location_similarity_weight=None, locality_approach='global', 
                 use_clustering=False, cluster_approach='user_based', clustering_method=None, n_clusters=None,
                 apply_temporal_similarity_in_calc=False, time_decay_function_type=None,
                 tau_exp=None, K_log=None, tau0_log=None, tau_pow=None,
                 temporal_locality_fusion_weight=None, 
                 min_rating=1, max_rating=5): # SUPPRIMÉ: user_locations, service_locations, user_timestamps
        
        self.k = k
        self.similarity_metric = similarity_metric
        self.approach = approach # 'user_based' ou 'item_based'
        self.normalize_ratings = normalize_ratings
        
        self.apply_location_similarity = apply_location_similarity
        self.location_similarity_weight = location_similarity_weight
        self.locality_approach = locality_approach # 'strong', 'custom', 'flexible', 'global'
        self.use_clustering = use_clustering
        self.cluster_approach = cluster_approach # 'user_based' ou 'service_based'
        self.clustering_method = clustering_method # 'kmeans' ou 'hierarchical'
        self.n_clusters = n_clusters
        
        self.apply_temporal_similarity_in_calc = apply_temporal_similarity_in_calc
        self.time_decay_function_type = time_decay_function_type
        self.tau_exp = tau_exp
        self.K_log = K_log
        self.tau0_log = tau0_log
        self.tau_pow = tau_pow
        self.temporal_locality_fusion_weight = temporal_locality_fusion_weight if temporal_locality_fusion_weight is not None else 0.5 # Default value
        
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.rt_matrix_3d = None # Sera défini dans fit()
        self.num_users = 0
        self.num_services = 0
        self.num_time_slices = 0
        self.user_means = {}
        self.service_means = {}
        self.global_mean = 0

        self.similarities_cache = {} # Cache pour les similarités précalculées
        self.neighbor_similarities_log = [] # Pour enregistrer les similarités utilisées dans la prédiction

        # SUPPRIMÉ: self.user_locations = user_locations
        # SUPPRIMÉ: self.service_locations = service_locations
        # SUPPRIMÉ: self.user_timestamps = user_timestamps

        # Avertissement si la similarité de localisation est activée sans données
        if self.apply_location_similarity:
            print("Avertissement: La similarité de localisation est activée, mais aucune donnée de localisation n'est fournie au modèle KNNClassique. Elle ne pourra pas être calculée.")
            self.apply_location_similarity = False # Désactive la similarité de localisation si les données ne sont pas là.
            self.location_similarity_weight = 0 # Annule tout poids
        
        # Initialisation pour le clustering de localisation (même si pas de données externes, la structure est là)
        self.user_clusters = None 
        self.service_clusters = None 

    def fit(self, rt_matrix_3d, precompute_all_similarities=True):
        """
        Initialise le modèle avec la matrice de temps de réponse 3D,
        calcule les moyennes et précalcule les similarités si nécessaire.
        """
        self.rt_matrix_3d = rt_matrix_3d
        self.num_users = rt_matrix_3d.shape[0]
        self.num_services = rt_matrix_3d.shape[1]
        self.num_time_slices = rt_matrix_3d.shape[2]

        self._calculate_user_means()
        self._calculate_service_means()
        self._get_global_mean_rating()

        if precompute_all_similarities:
            print("Précalcul des similarités...")
            if self.approach == 'user_based':
                for i in tqdm(range(self.num_users), desc="Similarités utilisateurs"):
                    for j in range(i + 1, self.num_users):
                        sim = self._calculate_similarity(i, j)
                        self.similarities_cache[(i, j)] = sim
                        self.similarities_cache[(j, i)] = sim # Symétrique
            elif self.approach == 'item_based':
                for i in tqdm(range(self.num_services), desc="Similarités services"):
                    for j in range(i + 1, self.num_services):
                        sim = self._calculate_similarity_items(i, j)
                        self.similarities_cache[(i, j)] = sim
                        self.similarities_cache[(j, i)] = sim # Symétrique
            print("Précalcul terminé.")

    def _get_global_mean_rating(self):
        """Calcule la moyenne globale des notes non nulles."""
        valid_ratings = self.rt_matrix_3d[self.rt_matrix_3d != 0]
        if valid_ratings.size > 0:
            self.global_mean = np.mean(valid_ratings)
        else:
            self.global_mean = (self.min_rating + self.max_rating) / 2 # Fallback
    
    def _calculate_user_means(self):
        """Calcule la moyenne des notes pour chaque utilisateur."""
        user_means = {}
        for u in range(self.num_users):
            user_ratings = self.rt_matrix_3d[u, :, :]
            valid_ratings = user_ratings[user_ratings != 0]
            if valid_ratings.size > 0:
                user_means[u] = np.mean(valid_ratings)
            else:
                user_means[u] = self.global_mean # Utilise la moyenne globale si aucune note

        self.user_means = user_means

    def _calculate_service_means(self):
        """Calcule la moyenne des notes pour chaque service."""
        service_means = {}
        for s in range(self.num_services):
            service_ratings = self.rt_matrix_3d[:, s, :]
            valid_ratings = service_ratings[service_ratings != 0]
            if valid_ratings.size > 0:
                service_means[s] = np.mean(valid_ratings)
            else:
                service_means[s] = self.global_mean # Utilise la moyenne globale si aucune note
        self.service_means = service_means

    def _get_user_ratings_for_similarity(self, user_id, target_time_slice=None):
        """
        Récupère les notes d'un utilisateur pour le calcul de similarité.
        Peut filtrer par tranche de temps si target_time_slice est fourni.
        Retourne un dictionnaire {item_id: rating}.
        """
        ratings_dict = {}
        for item_id in range(self.num_services):
            if target_time_slice is not None:
                rating = self.rt_matrix_3d[user_id, item_id, target_time_slice]
                if rating != 0:
                    ratings_dict[item_id] = rating
            else:
                # Si pas de target_time_slice, considérer toutes les notes non nulles
                # pour un item donné à travers toutes les tranches de temps
                item_ratings_at_all_times = self.rt_matrix_3d[user_id, item_id, :]
                valid_ratings_for_item = item_ratings_at_all_times[item_ratings_at_all_times != 0]
                if valid_ratings_for_item.size > 0:
                    # Utilise la moyenne des notes pour cet item à travers le temps
                    ratings_dict[item_id] = np.mean(valid_ratings_for_item)
        return ratings_dict

    def _get_item_ratings_for_similarity(self, item_id):
        """
        Récupère les notes d'un service (item) pour le calcul de similarité.
        Retourne un dictionnaire {user_id: rating}.
        """
        ratings_dict = {}
        for user_id in range(self.num_users):
            item_ratings_at_all_times = self.rt_matrix_3d[user_id, item_id, :]
            valid_ratings_for_user_item = item_ratings_at_all_times[item_ratings_at_all_times != 0]
            if valid_ratings_for_user_item.size > 0:
                ratings_dict[user_id] = np.mean(valid_ratings_for_user_item)
        return ratings_dict

    def _calculate_pearson_similarity(self, ratings1, ratings2, mean1, mean2):
        """Calcule la similarité de Pearson."""
        common_items = list(set(ratings1.keys()) & set(ratings2.keys()))
        if not common_items:
            return 0

        numerator = 0
        sum1_sq = 0
        sum2_sq = 0

        for item in common_items:
            diff1 = ratings1[item] - mean1
            diff2 = ratings2[item] - mean2
            numerator += diff1 * diff2
            sum1_sq += diff1**2
            sum2_sq += diff2**2

        denominator = math.sqrt(sum1_sq) * math.sqrt(sum2_sq)
        return numerator / denominator if denominator != 0 else 0

    def _calculate_euclidean_similarity(self, ratings1, ratings2):
        """Calcule la similarité euclidienne (distance, inversée pour similarité)."""
        common_items = list(set(ratings1.keys()) & set(ratings2.keys()))
        if not common_items:
            return 0

        sum_of_squares = sum([(ratings1[item] - ratings2[item])**2 for item in common_items])
        
        # Inverser la distance pour obtenir une similarité (plus la distance est petite, plus la similarité est grande)
        # Ajout d'une petite constante pour éviter la division par zéro et normalisation
        return 1 / (1 + math.sqrt(sum_of_squares))

    def _calculate_cosine_similarity(self, ratings1, ratings2):
        """Calcule la similarité cosinus."""
        common_items = list(set(ratings1.keys()) & set(ratings2.keys()))
        if not common_items:
            return 0

        dot_product = sum([ratings1[item] * ratings2[item] for item in common_items])
        magnitude1 = math.sqrt(sum([ratings1[item]**2 for item in ratings1.keys()]))
        magnitude2 = math.sqrt(sum([ratings2[item]**2 for item in ratings2.keys()]))

        denominator = magnitude1 * magnitude2
        return dot_product / denominator if denominator != 0 else 0

    def _get_latest_active_time_slice(self, entity_id, entity_type='user'):
        """
        Déduit la dernière tranche de temps active pour un utilisateur ou un service
        à partir de la matrice 3D.
        """
        if not hasattr(self, 'rt_matrix_3d') or self.rt_matrix_3d is None:
            raise AttributeError("rt_matrix_3d n'est pas initialisé. Appelez fit() avant de calculer les timestamps.")
        
        if entity_type == 'user':
            # Pour un utilisateur, on cherche les tranches de temps où il a noté au moins un service
            entity_ratings_all_times = self.rt_matrix_3d[entity_id, :, :] # (num_services, num_time_slices)
            active_time_indices = np.where(np.any(entity_ratings_all_times != 0, axis=0))[0]
        elif entity_type == 'service':
            # Pour un service, on cherche les tranches de temps où il a été noté par au moins un utilisateur
            entity_ratings_all_times = self.rt_matrix_3d[:, entity_id, :] # (num_users, num_time_slices)
            active_time_indices = np.where(np.any(entity_ratings_all_times != 0, axis=0))[0]
        else:
            raise ValueError("entity_type doit être 'user' ou 'service'.")

        if len(active_time_indices) > 0:
            return np.max(active_time_indices)
        return 0 # Si aucune activité, retourne 0 (ou une autre valeur par défaut appropriée)

    def _calculate_time_decay_factor(self, delta_t):
        """Calcule le facteur de décroissance temporelle."""
        if self.time_decay_function_type == 'exponential':
            # f(delta_t) = exp(-tau * delta_t)
            if self.tau_exp is None:
                raise ValueError("tau_exp doit être défini pour la fonction de décroissance exponentielle.")
            return math.exp(-self.tau_exp * delta_t)
        elif self.time_decay_function_type == 'logarithmic':
            # f(delta_t) = 1 / (1 + K * log(1 + delta_t / tau0))
            if self.K_log is None or self.tau0_log is None:
                raise ValueError("K_log et tau0_log doivent être définis pour la fonction de décroissance logarithmique.")
            return 1 / (1 + self.K_log * math.log(1 + delta_t / self.tau0_log))
        elif self.time_decay_function_type == 'power':
            # f(delta_t) = (1 + delta_t)^(-tau_pow)
            if self.tau_pow is None:
                raise ValueError("tau_pow doit être défini pour la fonction de décroissance de puissance.")
            return (1 + delta_t)**(-self.tau_pow)
        else:
            return 1.0 # Pas de décroissance si le type n'est pas reconnu

    def _get_location_similarity(self, user_i_idx, user_j_idx, entity_type='user'):
        """
        Calcule la similarité de localisation.
        Puisque les données de localisation ne sont plus passées explicitement,
        cette fonction ne peut pas effectuer un calcul basé sur des coordonnées géographiques.
        Elle retournera une similarité neutre (1.0) ou 0.0 si la similarité de localisation est activée
        mais que les données sont absentes.
        """
        # Si apply_location_similarity est False (définie dans __init__ si pas de données),
        # ou si nous n'avons pas les attributs nécessaires pour le calcul,
        # la similarité de localisation ne peut pas être appliquée.
        if not self.apply_location_similarity:
            return 1.0 # Valeur neutre pour la similarité (pas d'influence)

        # Si nous arrivons ici, cela signifie que apply_location_similarity était True lors de l'initialisation,
        # mais comme nous n'avons pas les données externes, nous ne pouvons pas calculer une vraie similarité de localisation.
        # Vous pourriez retourner 0.0 pour signifier une absence de similarité de localisation réelle,
        # ou 1.0 si vous voulez qu'elle ait un effet neutre sur la fusion.
        # Retournons 1.0 pour un effet neutre sur la fusion si la feature est "active" mais pas calculable.
        return 1.0 

    def _calculate_similarity(self, user_i_idx, user_j_idx, target_time_slice=None):
        """
        Calcule la similarité globale entre deux utilisateurs (ou éléments)
        en appliquant la métrique de base, puis en intégrant la similarité temporelle
        et de localisation via la formule de fusion si activée.
        """
        # 1. Calcul de la similarité de base (Pearson, Euclidienne, Cosinus)
        ratings1 = self._get_user_ratings_for_similarity(user_i_idx, target_time_slice)
        ratings2 = self._get_user_ratings_for_similarity(user_j_idx, target_time_slice)
        mean1 = self.user_means.get(user_i_idx, 0) if self.normalize_ratings else 0
        mean2 = self.user_means.get(user_j_idx, 0) if self.normalize_ratings else 0

        if self.similarity_metric == 'pearson':
            base_similarity = self._calculate_pearson_similarity(ratings1, ratings2, mean1, mean2)
        elif self.similarity_metric == 'euclidean':
            base_similarity = self._calculate_euclidean_similarity(ratings1, ratings2)
        elif self.similarity_metric == 'cosine':
            base_similarity = self._calculate_cosine_similarity(ratings1, ratings2)
        else:
            raise ValueError(f"Métrique de similarité inconnue: {self.similarity_metric}")

        final_similarity = base_similarity

        # 2. Calcul des scores de similarité temporelle et de localisation
        sim_temporal_score = base_similarity 
        if self.apply_temporal_similarity_in_calc and self.time_decay_function_type is not None:
            # Inférez les timestamps à partir de rt_matrix_3d
            ts1 = self._get_latest_active_time_slice(user_i_idx, entity_type='user') 
            ts2 = self._get_latest_active_time_slice(user_j_idx, entity_type='user')

            # Vérifications de type défensives (conservées)
            if not isinstance(ts1, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"Erreur de type: 'ts1' (timestamp de l'utilisateur {user_i_idx}) "
                    f"n'est pas numérique ({type(ts1)}: {ts1}). "
                    f"Vérifiez l'inférence des timestamps à partir de rt_matrix_3d."
                )
            if not isinstance(ts2, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"Erreur de type: 'ts2' (timestamp de l'utilisateur {user_j_idx}) "
                    f"n'est pas numérique ({type(ts2)}: {ts2}). "
                    f"Vérifiez l'inférence des timestamps à partir de rt_matrix_3d."
                )

            delta_t_users = abs(ts1 - ts2)
            
            time_decay_factor = self._calculate_time_decay_factor(delta_t_users)
            sim_temporal_score = base_similarity * time_decay_factor
               
            sim_temporal_score = max(-1.0, min(1.0, sim_temporal_score))
            
        sim_locality_score = base_similarity 
        # Si apply_location_similarity est True, mais qu'aucune donnée de localisation n'est fournie (géré dans __init__),
        # cette branche ne sera pas exécutée ou _get_location_similarity retournera une valeur neutre.
        if self.apply_location_similarity:
            sim_locality_score = self._get_location_similarity(user_i_idx, user_j_idx, entity_type='user')
            sim_locality_score = max(0.0, min(1.0, sim_locality_score))
        
        # 3. Logique de fusion principale
        l = self.temporal_locality_fusion_weight
        if not isinstance(l, (int, float, np.integer, np.floating)):
            print(f"Avertissement: temporal_locality_fusion_weight est de type {type(l)}. Attendu numérique. Utilisation de 0.5 par défaut.")
            l = 0.5 
        
        if self.apply_temporal_similarity_in_calc and self.apply_location_similarity:
            final_similarity = (l * sim_temporal_score) + ((1 - l) * sim_locality_score)
        elif self.apply_temporal_similarity_in_calc:
            final_similarity = sim_temporal_score 
        elif self.apply_location_similarity:
            final_similarity = sim_locality_score
        else:
            final_similarity = base_similarity
        
        return max(-1.0, min(1.0, final_similarity))

    def predict(self, user_id, item_id, time_slice):
        """
        Prédit la note pour un utilisateur, un élément et une tranche de temps donnés.
        """
        if self.rt_matrix_3d[user_id, item_id, time_slice] != 0:
            return self.rt_matrix_3d[user_id, item_id, time_slice] 

        target_user_mean = self.user_means.get(user_id, (self.min_rating + self.max_rating) / 2)

        candidate_neighbors = [u for u in range(self.num_users) if u != user_id and np.any(self.rt_matrix_3d[u, item_id, :] != 0)]
        
        if not candidate_neighbors:
            return target_user_mean 

        similarities = []
        for neighbor_user in candidate_neighbors:
            sim = self._calculate_similarity(user_id, neighbor_user, target_time_slice=time_slice)
            if sim is not None and sim > 0:
                similarities.append((neighbor_user, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_neighbors = similarities[:self.k]

        if not nearest_neighbors:
            return target_user_mean 

        numerator = 0.0
        denominator = 0.0

        for neighbor_user, sim in nearest_neighbors:
            self.neighbor_similarities_log.append({
                "target_user_id": user_id,
                "target_item_id": item_id,
                "target_time_slice": time_slice,
                "neighbor_id": neighbor_user,
                "similarity": sim
            })

            neighbor_item_ratings_at_times = self.rt_matrix_3d[neighbor_user, item_id, :]
            known_neighbor_item_ratings_indices = np.where(neighbor_item_ratings_at_times != 0)[0]
            
            if len(known_neighbor_item_ratings_indices) == 0:
                continue 

            relevant_timestamps = []
            if self.time_lookback is not None and self.time_lookback > 0: # Assuming time_lookback is an attribute of KNNClassique
                for t_idx in known_neighbor_item_ratings_indices:
                    if t_idx <= time_slice and t_idx >= (time_slice - self.time_lookback):
                        relevant_timestamps.append(t_idx)
            else: 
                relevant_timestamps = list(known_neighbor_item_ratings_indices)

            if not relevant_timestamps:
                continue 

            latest_relevant_timestamp = max(relevant_timestamps)
            neighbor_rating = self.rt_matrix_3d[neighbor_user, item_id, latest_relevant_timestamp]
            
            neighbor_mean = self.user_means.get(neighbor_user, 0)

            # Vérifications de type (conservées)
            if not isinstance(neighbor_rating, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"Erreur de type: 'neighbor_rating' pour l'utilisateur {neighbor_user}, "
                    f"l'élément {item_id}, la tranche {latest_relevant_timestamp} "
                    f"n'est pas numérique ({type(neighbor_rating)}: {neighbor_rating}). "
                    f"Vérifiez l'intégrité de rt_matrix_3d."
                )
            if not isinstance(neighbor_mean, (int, float, np.integer, np.floating)):
                raise TypeError(
                    f"Erreur de type: 'neighbor_mean' pour l'utilisateur {neighbor_user} "
                    f"n'est pas numérique ({type(neighbor_mean)}: {neighbor_mean}). "
                    f"Vérifiez le calcul ou le stockage de user_means."
                )

            if self.normalize_ratings:
                numerator += sim * (neighbor_rating - neighbor_mean)
            else:
                numerator += sim * neighbor_rating
            denominator += abs(sim)

        if denominator == 0:
            return target_user_mean

        if self.normalize_ratings:
            predicted_rating = target_user_mean + (numerator / denominator)
        else:
            predicted_rating = numerator / denominator

        return max(self.min_rating, min(self.max_rating, predicted_rating))

    def save_similarities_to_json(self, filepath):
        """Sauvegarde les similarités précalculées dans un fichier JSON."""
        try:
            # Convertir les clés de tuple en chaînes pour la sérialisation JSON
            serializable_cache = {str(k): v for k, v in self.similarities_cache.items()}
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_cache, f, indent=4, ensure_ascii=False)
            print(f"Similarités sauvegardées dans : {filepath}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des similarités : {e}")

    # Vous devrez probablement adapter _calculate_similarity_items si vous utilisez l'approche item-based
    # et qu'elle dépendait de user_timestamps ou user_locations.
    def _calculate_similarity_items(self, item_i_idx, item_j_idx):
        """Calcule la similarité entre deux items (pour l'approche item-based)."""
        ratings1 = self._get_item_ratings_for_similarity(item_i_idx)
        ratings2 = self._get_item_ratings_for_similarity(item_j_idx)
        mean1 = self.service_means.get(item_i_idx, 0) if self.normalize_ratings else 0
        mean2 = self.service_means.get(item_j_idx, 0) if self.normalize_ratings else 0

        if self.similarity_metric == 'pearson':
            return self._calculate_pearson_similarity(ratings1, ratings2, mean1, mean2)
        elif self.similarity_metric == 'euclidean':
            return self._calculate_euclidean_similarity(ratings1, ratings2)
        elif self.similarity_metric == 'cosine':
            return self._calculate_cosine_similarity(ratings1, ratings2)
        else:
            return 0 # Ou lever une erreur

    # Les méthodes de clustering (si elles dépendent de user_locations) devront être revues.
    # Pour l'instant, elles sont laissées telles quelles mais ne seront pas fonctionnelles sans données.
    def _perform_clustering(self):
        """Effectue le clustering des utilisateurs ou des services."""
        if not self.use_clustering:
            return

        if self.cluster_approach == 'user_based':
            # Impossible de clusteriser les utilisateurs sans leurs coordonnées de localisation.
            # Si self.user_locations n'est pas fourni, le clustering ne peut pas avoir lieu.
            if self.user_locations is None:
                print("Avertissement: Clustering basé sur les utilisateurs activé, mais self.user_locations est None. Le clustering ne sera pas effectué.")
                return

            data_to_cluster = np.array(list(self.user_locations.values()))
            if len(data_to_cluster) < self.n_clusters:
                print(f"Avertissement: Moins de points de données ({len(data_to_cluster)}) que de clusters ({self.n_clusters}) demandés pour le clustering utilisateur. Ajustement du nombre de clusters.")
                self.n_clusters = len(data_to_cluster) if len(data_to_cluster) > 0 else 1
                if self.n_clusters == 0: return

            if self.clustering_method == 'kmeans':
                cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
            elif self.clustering_method == 'hierarchical':
                cluster_model = AgglomerativeClustering(n_clusters=self.n_clusters)
            else:
                raise ValueError(f"Méthode de clustering inconnue: {self.clustering_method}")

            user_ids = list(self.user_locations.keys())
            labels = cluster_model.fit_predict(data_to_cluster)
            self.user_clusters = {user_ids[i]: labels[i] for i in range(len(user_ids))}

        elif self.cluster_approach == 'service_based':
            # Impossible de clusteriser les services sans leurs coordonnées de localisation.
            if self.service_locations is None:
                print("Avertissement: Clustering basé sur les services activé, mais self.service_locations est None. Le clustering ne sera pas effectué.")
                return
            
            data_to_cluster = np.array(list(self.service_locations.values()))
            if len(data_to_cluster) < self.n_clusters:
                print(f"Avertissement: Moins de points de données ({len(data_to_cluster)}) que de clusters ({self.n_clusters}) demandés pour le clustering service. Ajustement du nombre de clusters.")
                self.n_clusters = len(data_to_cluster) if len(data_to_cluster) > 0 else 1
                if self.n_clusters == 0: return

            if self.clustering_method == 'kmeans':
                cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
            elif self.clustering_method == 'hierarchical':
                cluster_model = AgglomerativeClustering(n_clusters=self.n_clusters)
            else:
                raise ValueError(f"Méthode de clustering inconnue: {self.clustering_method}")

            service_ids = list(self.service_locations.keys())
            labels = cluster_model.fit_predict(data_to_cluster)
            self.service_clusters = {service_ids[i]: labels[i] for i in range(len(service_ids))}

    def _get_cluster(self, entity_id, entity_type='user'):
        """Retourne le cluster d'une entité."""
        if entity_type == 'user' and self.user_clusters:
            return self.user_clusters.get(entity_id)
        elif entity_type == 'service' and self.service_clusters:
            return self.service_clusters.get(entity_id)
        return None

# --- Classe KNNEvaluator ---

class KNNEvaluator:
    def __init__(self, knn_model):
        self.knn_model = knn_model
        self.min_rating = self.knn_model.min_rating
        self.max_rating = self.knn_model.max_rating
        self.original_data = None
        self.masked_data = None
        self.test_data_coords = [] 
        self.user_timestamps = None # Sera passé au knn_model.fit

    def prepare_test_data(self, rt_matrix_3d, mask_percentage=0.1, random_seed=42, user_timestamps=None):
        """
        Masque un pourcentage des notes connues de la matrice 3D pour créer des données de test.
        """
        self.original_data = np.copy(rt_matrix_3d)
        self.masked_data = np.copy(rt_matrix_3d)
        self.test_data_coords = []
        self.user_timestamps = user_timestamps # Stocker les timestamps

        np.random.seed(random_seed)

        known_coords = np.argwhere(rt_matrix_3d != 0)
        np.random.shuffle(known_coords)

        num_to_mask = int(len(known_coords) * mask_percentage)
        
        for i in range(num_to_mask):
            u, item, t = known_coords[i]
            original_rating = self.masked_data[u, item, t]
            self.test_data_coords.append((u, item, t, original_rating))
            self.masked_data[u, item, t] = 0 

    def evaluate(self):
        """
        Évalue le modèle KNN basé sur les données masquées.
        Retourne MAE, RMSE et NRMSE.
        """
        if self.masked_data is None or not self.test_data_coords:
            raise ValueError("Les données de test n'ont pas été préparées. Appelez prepare_test_data en premier.")

        # KNNClassique.fit ne prend plus user_locations en paramètre direct, car il les calcule lui-même
        self.knn_model.fit(self.masked_data)

        predictions = []
        true_ratings = []

        for u, i, t, true_rating in tqdm(self.test_data_coords, total=len(self.test_data_coords), desc="Évaluation des prédictions"):
            predicted_rating = self.knn_model.predict(u, i, t)
            
            predictions.append(predicted_rating)
            true_ratings.append(true_rating)

        predictions = np.array(predictions)
        true_ratings = np.array(true_ratings)

        mae = mean_absolute_error(true_ratings, predictions)
        rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
        
        rating_range = self.max_rating - self.min_rating
        nrmse = rmse / rating_range if rating_range > 0 else 0

        return mae, rmse, nrmse

# --- Bloc principal ---

if __name__ == "__main__":
    np.random.seed(42)
    num_users = 20  
    num_items = 15  
    num_time_slices = 10 

    dataset = MockDataset(num_users, num_items, num_time_slices)

    evaluation_results_list = []

    print("--- Début de l'évaluation des modèles KNN ---")

    k_values_to_test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for k_val in k_values_to_test:
        print(f"\n--- Évaluation du modèle KNN Classique (k={k_val}) ---")
        params_basic_knn = {
            'k': k_val,
            'similarity_metric': 'pearson',
            'approach': 'user_based',
            'time_decay_function_type': None,
            'time_lookback': 0,
            'tau_exp': 1.0, 'K_log': 1.0, 'tau0_log': 0.5, 'tau_pow': 2.0, 
            'apply_temporal_similarity_in_calc': False,
            # Paramètres de localisation (désactivés pour ce modèle basique)
            'apply_location_similarity': False,
            'location_similarity_weight': 0.5,
            'location_decay_factor': 1.0, 
            'locality_approach': None,
            'location_clustering_type': None,
            'num_location_clusters': 0
        }
        print(f"Paramètres: {params_basic_knn}")
        knn_model = KNNClassique(**params_basic_knn)
        evaluator = KNNEvaluator(knn_model) 
        # user_locations n'est plus passé ici directement
        evaluator.prepare_test_data(dataset.rt_matrix_3d, mask_percentage=0.1, random_seed=42, 
                                    user_timestamps=dataset.user_timestamps) 
        mae, rmse, nrmse = evaluator.evaluate()
        
        results_current_k = {**params_basic_knn, 'MAE': mae, 'RMSE': rmse, 'NRMSE': nrmse}
        evaluation_results_list.append(results_current_k)
        print(f"Résultats: MAE={mae:.4f}, RMSE={rmse:.4f}, NRMSE={nrmse:.4f}")
        print("-" * 50)

    print("\n--- Évaluation du modèle combiné (Pénalisation exponentielle + Localité forte basée sur clustering) ---")
    params_combined_model = {
        'k': 3, 
        'similarity_metric': 'pearson',
        'approach': 'user_based',
        
        # Paramètres de Pénalisation Temporelle
        'apply_temporal_similarity_in_calc': True, 
        'time_decay_function_type': 'exponential', 
        'time_lookback': 2,                        
        'tau_exp': 1.0,                            
        'K_log': 1.0, 'tau0_log': 0.5, 'tau_pow': 2.0,
        
        # Paramètres de Localisation (basée sur les vecteurs abstraits user/service/temps)
        'apply_location_similarity': True,
        'location_similarity_weight': 0.5,
        'location_decay_factor': 1.0, # Peut ajuster la sensibilité de la distance abstraite
        'locality_approach': 'flexible', # Changé en flexible pour montrer l'utilisation de la distance sur vecteurs abstraits
        'location_clustering_type': 'kmeans',      
        'num_location_clusters': 2                 
    }
    print(f"Paramètres: {params_combined_model}")
    knn_model_combined = KNNClassique(**params_combined_model)
    evaluator_combined = KNNEvaluator(knn_model_combined)
    # user_locations n'est plus passé ici directement
    evaluator_combined.prepare_test_data(dataset.rt_matrix_3d, mask_percentage=0.1, random_seed=42, 
                                        user_timestamps=dataset.user_timestamps) 
    mae_combined, rmse_combined, nrmse_combined = evaluator_combined.evaluate()
    
    results_combined_model = {**params_combined_model, 'MAE': mae_combined, 'RMSE': rmse_combined, 'NRMSE': nrmse_combined}
    evaluation_results_list.append(results_combined_model)
    print(f"Résultats: MAE={mae_combined:.4f}, RMSE={rmse_combined:.4f}, NRMSE={nrmse_combined:.4f}")
    print("-" * 50)

    print("\n--- Tableau récapitulatif des évaluations ---")
    evaluation_table_df = pd.DataFrame(evaluation_results_list)
    
    evaluation_table_df = evaluation_table_df.fillna('')
    
    evaluation_table_df = evaluation_table_df.sort_values(by='RMSE').reset_index(drop=True)
    
    print(evaluation_table_df.to_string())

    print("\n--- Configuration du meilleur modèle (basé sur le RMSE le plus bas) ---")
    if not evaluation_table_df.empty:
        best_row = evaluation_table_df.iloc[0]
        print(best_row.to_string())
    else:
        print("Aucun modèle valide n'a été évalué.")