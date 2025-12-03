# --- 1. Importações ---
import pandas as pd
import joblib
import shap
import xgboost
import math
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from similarity_calculator import SimilarityFinder

print("Iniciando o servidor e carregando os recursos...")
app = Flask(__name__)
CORS(app)

def clean_nans(data_dict):
    for key, value in data_dict.items():
        if isinstance(value, float) and math.isnan(value): data_dict[key] = None
    return data_dict

try:
    model = joblib.load('../modelo_xgboost.pkl')
    explainer = joblib.load('../shap_explainer.pkl')
    df_features = pd.read_csv('../dataset_reduzido_renomeadas2.csv')
    df_info = pd.read_csv('../../escolas_com_cep.csv')
    df_sim_dif_notas = pd.read_csv('../escola_similar_notas_diferentes.csv')
    df_sim_mesma_nota = pd.read_csv('../escola_diferentes_notas_similar.csv')
    
    df_similaridade_total = pd.concat([df_sim_dif_notas, df_sim_mesma_nota])

    df_master = pd.merge(df_info, df_features, on='ID_ESCOLA', how='inner')
    
    FEATURES = model.get_booster().feature_names

    escolas_ids_a_remover = [26035553, 29387019, 31000311, 43040918, 21243875, 26027836, 13001345, 21315604]
    df_para_similaridade = df_master[~df_master['ID_ESCOLA'].isin(escolas_ids_a_remover)]
    similarity_finder = SimilarityFinder(df_para_similaridade, FEATURES)

    print("Pré-calculando todos os valores SHAP para otimização...")
    df_master_features = df_master.set_index('ID_ESCOLA')[FEATURES].fillna(0)
    all_shap_values = explainer(df_master_features)
    df_shap_values = pd.DataFrame(all_shap_values.values, columns=FEATURES, index=df_master_features.index)
    
    print("Todos os recursos foram carregados. Servidor pronto!")
except Exception as e:
    print(f"Ocorreu um erro durante a inicialização: {e}")
    exit()

# --- ENDPOINT DO GRÁFICO GERAL (MODIFICADO) ---
@app.route('/api/shap_summary', methods=['GET'])
def get_shap_summary():
    try:
        N_FEATURES = 15
        N_SAMPLES = 500 # Amostra para não sobrecarregar o navegador

        mean_abs_shap = df_shap_values.abs().mean().sort_values(ascending=False)
        top_features = mean_abs_shap.head(N_FEATURES).index.tolist()

        plot_data = []
        
        num_schools = len(df_shap_values)
        sample_indices = np.random.choice(
            df_shap_values.index, 
            min(N_SAMPLES, num_schools), 
            replace=False
        )
        
        sampled_shap = df_shap_values.loc[sample_indices]
        sampled_features = df_master_features.loc[sample_indices]
        
        # --- MUDANÇA: Buscar os nomes das escolas amostradas ---
        school_names_map = df_master.set_index('ID_ESCOLA')['NO_ENTIDADE']
        sampled_school_names = school_names_map.loc[sample_indices] # É uma Series
        # --- FIM DA MUDANÇA ---

        top_features.reverse()

        # Prepara o payload para o gráfico
        for i, feature in enumerate(top_features):
            shap_vals_series = sampled_shap[feature]
            feature_vals_series = sampled_features[feature]
            
            min_val, max_val = feature_vals_series.min(), feature_vals_series.max()
            normalized_vals_series = (feature_vals_series - min_val) / (max_val - min_val) if max_val > min_val else pd.Series(0, index=feature_vals_series.index)

            # --- MUDANÇA: Iterar sobre o índice (ID_ESCOLA) para pegar o nome ---
            for school_id in shap_vals_series.index:
                shap_val = shap_vals_series.loc[school_id]
                norm_val = normalized_vals_series.loc[school_id]
                school_name = sampled_school_names.loc[school_id] # Pega o nome

                plot_data.append({
                    "x": shap_val,
                    "y": i, 
                    "normalized_value": norm_val,
                    "school_name": school_name # <-- ADICIONADO
                })
            # --- FIM DA MUDANÇA ---

        return jsonify({
            "features": top_features,
            "plot_data": plot_data
        })
    except Exception as e:
        print(f"Erro em get_shap_summary: {e}")
        return jsonify({"error": "Erro ao gerar o resumo SHAP"}), 500

@app.route('/api/escolas', methods=['GET'])
def get_escolas():
    escolas_list = df_master[['ID_ESCOLA', 'NO_ENTIDADE']].dropna().to_dict('records')
    return jsonify(escolas_list)

@app.route('/api/escola_dashboard/<int:escola_id>', methods=['GET'])
def get_escola_dashboard(escola_id):
    escola_data = df_master[df_master['ID_ESCOLA'] == escola_id]
    if escola_data.empty: return jsonify({"error": "Escola não encontrada"}), 404
    
    escola_features = escola_data[FEATURES].fillna(0)
    predicao_original = model.predict(escola_features)[0]
    
    shap_values_series = df_shap_values.loc[escola_id]
    
    shap_data_formatted = []
    for f, v in shap_values_series.items():
        feature_value = escola_data[f].iloc[0]
        shap_data_formatted.append({"feature": f, "shap_value": v, "feature_value": feature_value})
    
    shap_data_formatted.sort(key=lambda x: x['shap_value'], reverse=False)
    
    similares_ids = df_similaridade_total[df_similaridade_total['ID_i'] == escola_id]['ID_j'].unique().tolist()
    
    escolas_similares_data = df_master[df_master['ID_ESCOLA'].isin(similares_ids)].head(8) 
    
    response = {
        "detalhes_escola": clean_nans(escola_data.to_dict('records')[0]),
        "dados_shap": shap_data_formatted,
        "escolas_similares": [clean_nans(s) for s in escolas_similares_data.to_dict('records')],
        "previsao_modelo_original": float(predicao_original)
    }
    return jsonify(response)

@app.route('/api/comparar_shap/<int:id_escola_1>/<int:id_escola_2>', methods=['GET'])
def comparar_shap(id_escola_1, id_escola_2):
    if id_escola_1 not in df_shap_values.index or id_escola_2 not in df_shap_values.index:
        return jsonify({"error": "Uma das escolas não foi encontrada"}), 404
        
    shap_values_1 = df_shap_values.loc[id_escola_1]
    shap_values_2 = df_shap_values.loc[id_escola_2]
    
    tabela_comparativa = [{"feature": f, "shap_escola_1": shap_values_1[f], "shap_escola_2": shap_values_2[f], "diferenca": shap_values_2[f] - shap_values_1[f]} for f in FEATURES]
    return jsonify(tabela_comparativa)

@app.route('/api/contrafactual', methods=['POST'])
def simular_contrafactual():
    dados = request.json
    escola_id = int(dados['escola_id'])
    vetor_original = df_master[df_master['ID_ESCOLA'] == escola_id][FEATURES].iloc[0].copy()
    for feature, value in dados['features'].items():
        if feature in vetor_original: vetor_original[feature] = value
    predicao = model.predict(vetor_original.to_frame().T)
    return jsonify({'media_simulada': float(predicao[0])})

@app.route('/api/encontrar_escolas_simuladas', methods=['POST'])
def encontrar_escolas_simuladas():
    data = request.json
    simulated_vector = data['features']
    context_features = data.get('context_features')
    similar_schools = similarity_finder.find_similar_to_vector(simulated_vector, n_results=5, context_features=context_features)
    return jsonify([clean_nans(s) for s in similar_schools])

@app.route('/api/ganho_features/<int:escola_id>', methods=['GET'])
def get_ganho_features(escola_id):
    if escola_id not in df_shap_values.index:
        return jsonify({"error": "Escola não encontrada"}), 404

    top_10_percent_cutoff = df_master['MEDIA_FINAL'].quantile(0.9)
    vizinhança_ids = df_master[df_master['MEDIA_FINAL'] >= top_10_percent_cutoff]['ID_ESCOLA'].values
    
    shap_vizinhança_medio = df_shap_values.loc[vizinhança_ids].mean()
    shap_escola_selecionada = df_shap_values.loc[escola_id]
    
    ganho = shap_vizinhança_medio - shap_escola_selecionada
    
    ganho_ordenado = ganho.sort_values(ascending=False).head(10)
    resultado = [{"feature": index, "ganho": value} for index, value in ganho_ordenado.items()]
    
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

