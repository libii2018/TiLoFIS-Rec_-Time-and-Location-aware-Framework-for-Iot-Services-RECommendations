import os
import json
import numpy as np 
from tqdm import tqdm 
import sys

from dataset.dataset2.dataset import WSDREAMDataset
from modeles.KNN.knnClassique import KNNClassique
from modeles.KNN.knnClassique import KNNEvaluator


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    dataset_choice = sys.argv[1]
    percentage_str = sys.argv[2]
    type_data_choice =sys.argv[3]
    type_modele = sys.argv[4]

    if dataset_choice == '1':
        dataset_path_segment = 'dataset/dataset1'
        print("Using Dataset 1")
    elif dataset_choice == '2':
        dataset_path_segment = 'dataset/dataset2'
        print("Using Dataset 2")
    else:
        print("Invalid dataset choice. Please use '1' or '2'.")
        sys.exit(1)

    data_directory_path = os.path.join(os.path.dirname(__file__), dataset_path_segment, '')

    if type_data_choice == '1':
        print("Temps de reponse")
        results_dir = "experiment_results/tpdata"
        matrice_dir="matrice_similarite/tpdata"
    elif type_data_choice == '2':
        print("Debit")
        results_dir = "experiment_results/rtdata"
        matrice_dir="matrice_similarite/rtdata"
    else:
        print("Invalid data type choice. Please use '1' or '2'.")
        sys.exit(1)

    os.makedirs(results_dir, exist_ok=True)
    evaluation_results_filepath = os.path.join(results_dir, "evaluation_results.json")

    try:
        percentage = float(percentage_str)
        if not (0 < percentage <= 100):
            raise ValueError("Percentage must be between 0 and 100.")
        percentage_for_dataset = percentage / 100.0

        dataset = WSDREAMDataset(local_dataset_path=data_directory_path, percentage=percentage)
        if type_data_choice == '1':
            dataset.load_tp_data_to_3d_matrix()
        elif type_data_choice == '2':
            dataset.load_rt_data_to_3d_matrix()
        else:
            print("Invalid data type choice. Please use '1' or '2'.")
            sys.exit(1)

        if dataset.rt_matrix_3d is not None:
            print(f"Forme de la matrice : {dataset.rt_matrix_3d.shape}")
            print("\n--- Test KNN User-based avec précalcul et sauvegarde ---")

            k_values_to_test = [1, 2, 3]
            apply_temporal_similarity_in_calc_options = [True]
            time_decay_function_type_options = ['exponential', 'logarithmic', 'power']
            time_decay_factor_options = [0.01, 0.05, 0.1, 0.2, 0.5]
            apply_location_similarity_options = [True]
            location_similarity_weight_options = [0.2, 0.5, 0.8] 
            locality_type_options = ['strong', 'custom', 'flexible']
            location_clustering_type_options = ['kmeans','hierarchical']
            num_location_clusters_options = [10, 30, 50] 
            temporal_locality_fusion_weight_options = [0.1, 0.5, 0.9] 

            evaluation_results_list = []
            
            # --- Experimentation KNN simple ---
            if type_modele == '1':
                print("\n--- Début de l'expérimentation KNN simple ---")
                for k_val in k_values_to_test:
                    params_basic_knn = {
                        'k': k_val,
                        'similarity_metric': 'pearson',
                        'approach': 'user_based',
                        'apply_location_similarity': False,
                        'apply_temporal_similarity_in_calc': False
                    }
                    knn_model = KNNClassique(**params_basic_knn)                    
                    knn_model.fit(dataset.rt_matrix_3d) 
                    evaluator = KNNEvaluator(knn_model) 
                    evaluator.prepare_test_data(dataset.rt_matrix_3d, mask_percentage=0.1, random_seed=42) 
                    mae, rmse, nrmse = evaluator.evaluate()                     
                    results_current_k = {**params_basic_knn, 'MAE': mae, 'RMSE': rmse, 'NRMSE': nrmse}
                    evaluation_results_list.append(results_current_k)
                    similarity_filename = f"similarities_knn_simple_k_{k_val}.json"
                    similarity_filepath = os.path.join(matrice_dir, similarity_filename)
                    if hasattr(knn_model, 'save_similarities_log_to_json'):
                        knn_model.save_similarities_log_to_json(similarity_filepath)
                    elif hasattr(knn_model, 'save_similarities_to_json'):
                        knn_model.save_similarities_to_json(similarity_filepath)
                    else:
                        print(f"Avertissement: Méthode de sauvegarde des similarités non trouvée pour k={k_val}.")

            # --- Expérimentation KNN introduction du temps ---
            elif type_modele == '2':
                print("\n--- Début de l'expérimentation KNN avec introduction du temps ---")
                for k_val in k_values_to_test: 
                    for apply_temporal in apply_temporal_similarity_in_calc_options:
                        if not apply_temporal: continue 
                        for decay_type in time_decay_function_type_options:
                            if decay_type is None: continue 
                            for decay_factor in time_decay_factor_options:
                                params_knn = {
                                    'k': k_val,
                                    'similarity_metric': 'pearson',
                                    'approach': 'user_based',
                                    'apply_location_similarity': False, 
                                    'apply_temporal_similarity_in_calc': apply_temporal,
                                    'time_decay_function_type': decay_type,
                                    'tau_exp': decay_factor if decay_type == 'exponential' else None,
                                    'K_log': decay_factor if decay_type == 'logarithmic' else None,
                                    'tau0_log': decay_factor / 10.0 if decay_type == 'logarithmic' else None, 
                                    'tau_pow': decay_factor if decay_type == 'power' else None
                                }
                                knn_model = KNNClassique(**params_knn)
                                knn_model.fit(dataset.rt_matrix_3d) 
                                evaluator = KNNEvaluator(knn_model) 
                                evaluator.prepare_test_data(
                                    dataset.rt_matrix_3d, 
                                    mask_percentage=0.1, 
                                    random_seed=42
                                ) 
                                mae, rmse, nrmse = evaluator.evaluate() 
                                results_current_run = {**params_knn, 'MAE': mae, 'RMSE': rmse, 'NRMSE': nrmse}
                                evaluation_results_list.append(results_current_run)
                                similarity_filename = f"similarities_knn_time_k_{k_val}_decay_{decay_type}_factor_{decay_factor}.json"
                                similarity_filepath = os.path.join(matrice_dir, similarity_filename)
                                if hasattr(knn_model, 'save_similarities_log_to_json'):
                                    knn_model.save_similarities_log_to_json(similarity_filepath)
                                elif hasattr(knn_model, 'save_similarities_to_json'):
                                    knn_model.save_similarities_to_json(similarity_filepath)
                                else:
                                    print(f"Avertissement: Méthode de sauvegarde des similarités non trouvée pour k={k_val}.")

            # --- Experimentation KNN introduction de la localisation ---
            elif type_modele == '3':
                print("\n--- Début de l'expérimentation KNN avec introduction de la localisation ---")
                for k_val in k_values_to_test: 
                    for current_location_weight in location_similarity_weight_options: 
                        for location_clustering_type in location_clustering_type_options: 
                            for num_location_clusters in num_location_clusters_options: 
                                for apply_location in apply_location_similarity_options:
                                    if not apply_location: continue 
                                    for locality_type in locality_type_options:                                    
                                        print(f"\n--- Évaluation du modèle KNN Classique (k={k_val}) avec Localisation ---")
                                        params_knn = {
                                            'k': k_val,
                                            'similarity_metric': 'pearson',
                                            'approach': 'user_based',
                                            'apply_location_similarity': apply_location,
                                            'location_similarity_weight': current_location_weight, 
                                            'locality_approach': locality_type, 
                                        }                                        
                                        knn_model = KNNClassique(**params_knn)
                                        knn_model.fit(dataset.rt_matrix_3d) 
                                        evaluator = KNNEvaluator(knn_model) 
                                        evaluator.prepare_test_data(
                                            dataset.rt_matrix_3d, 
                                            mask_percentage=0.1, 
                                            random_seed=42
                                        ) 
                                        mae, rmse, nrmse = evaluator.evaluate() 
                                        results_current_run = {**params_knn, 'MAE': mae, 'RMSE': rmse, 'NRMSE': nrmse}
                                        evaluation_results_list.append(results_current_run)
                                        weight_str = str(current_location_weight).replace('.', '_')
                                        similarity_filename = f"similarities_knn_location_k_{k_val}_loc_cluster_{location_clustering_type}_{num_location_clusters}_weight_{weight_str}_approach_{locality_type}.json"
                                        similarity_filepath = os.path.join(matrice_dir, similarity_filename)
                                        if hasattr(knn_model, 'save_similarities_log_to_json'):
                                            knn_model.save_similarities_log_to_json(similarity_filepath)
                                        elif hasattr(knn_model, 'save_similarities_to_json'):
                                            knn_model.save_similarities_to_json(similarity_filepath)
                                        else:
                                            print(f"Avertissement: Méthode de sauvegarde des similarités non trouvée pour k={k_val}.")

            # --- KNN introduction temps et localisation ---    
            elif type_modele == '4':     
                print("\n--- Début de l'expérimentation KNN avec introduction du temps et de la localisation ---")  
                for k_val in k_values_to_test: 
                    for current_location_weight in location_similarity_weight_options: 
                        for location_clustering_type in location_clustering_type_options: 
                            for num_location_clusters in num_location_clusters_options: 
                                if num_location_clusters > 0 and num_location_clusters > dataset.rt_matrix_3d.shape[0]: 
                                    continue
                                for apply_location in apply_location_similarity_options:
                                    if not apply_location: continue 
                                    for locality_type in locality_type_options:
                                        for apply_temporal in apply_temporal_similarity_in_calc_options:
                                            if not apply_temporal: continue 
                                            for decay_type in time_decay_function_type_options:
                                                if decay_type is None: continue 
                                                for decay_factor in time_decay_factor_options:
                                                    for fusion_weight in temporal_locality_fusion_weight_options: 
                                                        print(f"\n--- Évaluation du modèle KNN Classique (k={k_val}) avec Temps et Localisation ---")
                                                        params_knn = {
                                                            'k': k_val,
                                                            'similarity_metric': 'pearson',
                                                            'approach': 'user_based',
                                                            'apply_location_similarity': True,
                                                            'location_similarity_weight': current_location_weight, 
                                                            'locality_approach': locality_type, 
                                                            'apply_temporal_similarity_in_calc': True, 
                                                            'time_decay_function_type': decay_type,
                                                            'tau_exp': decay_factor if decay_type == 'exponential' else None,
                                                            'K_log': decay_factor if decay_type == 'logarithmic' else None,
                                                            'tau0_log': decay_factor / 10.0 if decay_type == 'logarithmic' else None, 
                                                            'tau_pow': decay_factor if decay_type == 'power' else None,
                                                            'temporal_locality_fusion_weight': fusion_weight 
                                                        }                                                        
                                                        knn_model = KNNClassique(**params_knn)
                                                        knn_model.fit(dataset.rt_matrix_3d) 
                                                        evaluator = KNNEvaluator(knn_model) 
                                                        evaluator.prepare_test_data(
                                                            dataset.rt_matrix_3d, 
                                                            mask_percentage=0.1, 
                                                            random_seed=42
                                                        )
                                                        mae, rmse, nrmse = evaluator.evaluate()
                                                        results_current_run = {**params_knn, 'MAE': mae, 'RMSE': rmse, 'NRMSE': nrmse}
                                                        evaluation_results_list.append(results_current_run)
                                                        weight_str = str(current_location_weight).replace('.', '_')
                                                        decay_factor_str = str(decay_factor).replace('.', '_')
                                                        fusion_weight_str = str(fusion_weight).replace('.', '_')
                                                        similarity_filename = (
                                                            f"similarities_knn_combined_k_{k_val}_decay_{decay_type}_factor_{decay_factor_str}_"
                                                            f"loc_cluster_{location_clustering_type}_{num_location_clusters}_weight_{weight_str}_"
                                                            f"approach_{locality_type}_fusion_{fusion_weight_str}.json"
                                                        )
                                                        similarity_filepath = os.path.join(matrice_dir, similarity_filename)
                                                        if hasattr(knn_model, 'save_similarities_log_to_json'):
                                                            knn_model.save_similarities_log_to_json(similarity_filepath)
                                                        elif hasattr(knn_model, 'save_similarities_to_json'):
                                                            knn_model.save_similarities_to_json(similarity_filepath)
                                                        else:
                                                            print(f"Avertissement: Méthode de sauvegarde des similarités non trouvée pour k={k_val}.")
                                                        
           
            try:
                with open(evaluation_results_filepath, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_results_list, f, indent=4, ensure_ascii=False)
                print(f"\nTous les résultats d'évaluation ont été sauvegardés dans : {evaluation_results_filepath}")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde des résultats d'évaluation JSON : {e}")

    except FileNotFoundError as e:
        print(f"Erreur : {e}. Assurez-vous que le dossier '{data_directory_path}' existe et contient les fichiers nécessaires.")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite : {e}")