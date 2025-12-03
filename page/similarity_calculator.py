# similarity_calculator.py

import pandas as pd
import numpy as np

class SimilarityFinder:
    """
    Uma classe para encontrar escolas reais que são similares a um vetor
    de características hipotético (simulado), usando distância de percentil.
    Pode operar em todas as features ou num subconjunto contextual.
    """
    def __init__(self, df_master, features_list):
        """
        Inicializa o buscador de similaridade.
        """
        print("Inicializando o SimilarityFinder com distância de percentil...")
        self.df_master = df_master.copy()
        self.features_list = list(features_list)
        self.all_schools_features = self.df_master[self.features_list].fillna(0)
        self.num_schools = len(self.all_schools_features)

        # Pré-calcula e armazena um dicionário com cada feature ordenada.
        self.sorted_features = {
            feature: np.sort(self.all_schools_features[feature].values)
            for feature in self.features_list
        }
        print("Features pré-ordenadas para cálculo de percentil.")

    def _calculate_percentile_distance(self, vector1_series, vector2_series, context_features):
        """
        Calcula a distância média de percentil entre dois vetores.
        """
        distances = []
        for feature in context_features:
            # Encontra a posição (ranking) de cada valor no array ordenado
            rank1 = np.searchsorted(self.sorted_features[feature], vector1_series.get(feature, 0))
            rank2 = np.searchsorted(self.sorted_features[feature], vector2_series.get(feature, 0))
            
            # Calcula a diferença de percentil
            percentile_diff = abs((rank1 / self.num_schools) - (rank2 / self.num_schools))
            distances.append(percentile_diff)
            
        return np.mean(distances) if distances else 0

    def find_similar_to_vector(self, simulated_vector, n_results=5, context_features=None):
        """
        Encontra as escolas mais próximas do vetor de características simulado.
        """
        if not context_features:
            context_features = self.features_list
            
        simulated_series = pd.Series(simulated_vector)

        # Calcula a distância do vetor simulado para cada escola
        distances = self.all_schools_features.apply(
            lambda row: self._calculate_percentile_distance(simulated_series, row, context_features),
            axis=1
        )

        # Obtém os índices das escolas com as menores distâncias
        closest_indices = distances.nsmallest(n_results).index

        similar_schools = self.df_master.loc[closest_indices].copy()
        
        # Adiciona a distância calculada (como percentagem) a cada escola
        similar_schools['distance'] = (distances[closest_indices] * 100)

        return similar_schools.to_dict('records')
