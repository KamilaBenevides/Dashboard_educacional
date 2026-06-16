# similarity_calculator.py

from typing import List, Dict, Any, Optional
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

    def find_best_cost_benefit_neighbors(
        self, 
        original_vector: pd.Series, 
        real_performance_series: pd.Series, 
        original_real_performance: float, 
        n_results: int = 5, 
        context_features: list = None
    ) -> list:
        """
        Encontra escolas vizinhas priorizando a melhor razão custo-benefício.
        O custo é a distância de percentil e o benefício é o crescimento na PERFORMANCE REAL.
        Fórmula: Distância / Crescimento Real. Menores scores são melhores.
        """
        if not context_features:
            context_features = self.features_list

        # Calcula a distância de percentil para todas as escolas presentes em all_schools_features
        distances: pd.Series = self.all_schools_features.apply(
            lambda row: self._calculate_percentile_distance(original_vector, row, context_features),
            axis=1
        )
        
        # Alinha a série de performance real com os índices válidos do buscador de similaridade
        valid_indices: pd.Index = distances.index
        aligned_real_perf: pd.Series = real_performance_series.loc[valid_indices]
        
        # Calcula o crescimento real (Benefício Real)
        benefits: pd.Series = aligned_real_perf - original_real_performance
        
        # Calcula o score bruto (Custo / Benefício)
        cb_score: pd.Series = distances / benefits
        
        # Mantém o score apenas se houver ganho real (benefits > 0.001), senão atribui Infinito (np.inf)
        cb_score = cb_score.where(benefits > 0.001, np.inf)
        
        # Obtém os índices com os menores scores
        raw_closest_indices: pd.Index = cb_score.nsmallest(n_results).index
        
        # Filtra os índices válidos removendo aqueles que resultaram em Infinito
        closest_indices: list = [idx for idx in raw_closest_indices if cb_score.loc[idx] != np.inf]
        
        if not closest_indices:
            return []
            
        similar_schools: pd.DataFrame = self.df_master.loc[closest_indices].copy()
        
        # Adiciona as métricas calculadas ao DataFrame de retorno
        similar_schools['distance'] = (distances.loc[closest_indices] * 100)
        similar_schools['expected_growth'] = benefits.loc[closest_indices]
        similar_schools['cb_score'] = cb_score.loc[closest_indices]
        
        return similar_schools.to_dict('records')