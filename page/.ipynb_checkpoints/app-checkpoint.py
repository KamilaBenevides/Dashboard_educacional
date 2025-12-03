# --- 1. Importações ---
# Importa as bibliotecas necessárias para o servidor web, manipulação de dados e machine learning.
import pandas as pd
import joblib
import shap
import xgboost
from flask import Flask, jsonify, request
from flask_cors import CORS

print("Iniciando o servidor e carregando os recursos...")

# --- 2. Inicialização do Aplicativo Flask ---
# Cria a instância principal do servidor web.
app = Flask(__name__)
# Habilita o CORS para permitir que a página web (frontend) acesse este servidor.
CORS(app)

# --- 3. Carregamento de Dados e Modelos ---
# Esta parte é executada apenas uma vez, quando o servidor é iniciado.

try:
    # Carrega o modelo de Machine Learning treinado.
    model = joblib.load('../modelo_xgboost.pkl')
    print("Modelo 'modelo_xgboost.pkl' carregado com sucesso.")

    # Carrega os dados de features e notas.
    df_features = pd.read_csv('../dataset_reduzido_renomeadas2.csv')
    print("Dataset de features 'dados_escolas.csv' carregado.")

    # Carrega os dados cadastrais das escolas.
    df_info = pd.read_csv('../../escolas_com_cep.csv')
    print("Dataset de informações 'escolas_com_cep.csv' carregado.")

    # Carrega os datasets de similaridade.
    df_sim_dif_notas = pd.read_csv('../escola_similar_notas_diferentes.csv')
    df_sim_mesma_nota = pd.read_csv('../escola_diferentes_notas_similar.csv')
    print("Datasets de similaridade carregados.")

    # --- 4. Preparação dos Dados ---
    # Combina as informações cadastrais com os dados de features em um único DataFrame.
    # Isso facilita a busca de informações completas de uma escola.
    df_master = pd.merge(df_info, df_features, on='ID_ESCOLA', how='inner')
    print("Merge entre datasets de features e informações concluído.")
    
    # Define as colunas que são features do modelo (excluindo identificadores e o alvo).
    # IMPORTANTE: Adapte esta lista se suas colunas de features forem diferentes.
    FEATURES = [col for col in df_features.columns if col not in ['ID_ESCOLA', 'MEDIA_FINAL']]
    
    # Prepara o explicador SHAP, que será usado para calcular a importância das features.
    explainer = shap.TreeExplainer(model)
    print("Explicador SHAP inicializado. Servidor pronto!")

except FileNotFoundError as e:
    print(f"Erro crítico: Arquivo não encontrado - {e}. Verifique se todos os arquivos .csv e .pkl estão na mesma pasta que app.py.")
    exit() # Encerra o script se um arquivo essencial não for encontrado.


# --- 5. Definição dos Endpoints da API ---
# Endpoints são as URLs que o frontend irá chamar para obter dados.

@app.route('/api/escolas', methods=['GET'])
def get_escolas():
    """
    Endpoint para listar todas as escolas disponíveis para seleção no frontend.
    Retorna uma lista de objetos, cada um com o ID e o nome da escola.
    """
    escolas_list = df_master[['ID_ESCOLA', 'NO_ENTIDADE']].dropna().to_dict('records')
    return jsonify(escolas_list)

@app.route('/api/escola_dashboard/<int:escola_id>', methods=['GET'])
def get_escola_dashboard(escola_id):
    """
    Endpoint principal que retorna um pacote de dados para o dashboard de uma escola.
    """
    # Encontra os dados da escola selecionada no DataFrame principal.
    escola_data = df_master[df_master['ID_ESCOLA'] == escola_id]
    if escola_data.empty:
        return jsonify({"error": "Escola não encontrada"}), 404

    # --- Cálculo SHAP em tempo real ---
    escola_features = escola_data[FEATURES]
    shap_values = explainer.shap_values(escola_features)
    
    # Formata os valores SHAP para o frontend (gráfico).
    shap_data_formatted = []
    for feature, shap_value in zip(FEATURES, shap_values[0]):
        shap_data_formatted.append({"feature": feature, "value": float(shap_value)})
    
    # Ordena por valor absoluto para o gráfico de cascata.
    shap_data_formatted.sort(key=lambda x: abs(x['value']), reverse=True)

    # --- Busca de Escolas Similares ---
    # Concatena os dois dataframes de similaridade para busca unificada.
    df_similaridade_total = pd.concat([df_sim_dif_notas, df_sim_mesma_nota])
    
    # Encontra os IDs das escolas similares à escola selecionada.
    similares_ids = df_similaridade_total[df_similaridade_total['ID_i'] == escola_id]['ID_j'].unique().tolist()
    
    # Busca as informações completas das escolas similares (limita a 4, como solicitado).
    escolas_similares_data = df_master[df_master['ID_ESCOLA'].isin(similares_ids)].head(4)
    
    # --- Monta a resposta final ---
    response = {
        "detalhes_escola": escola_data.to_dict('records')[0],
        "dados_shap": shap_data_formatted,
        "escolas_similares": escolas_similares_data.to_dict('records')
    }
    return jsonify(response)


@app.route('/api/comparar_shap/<int:id_escola_1>/<int:id_escola_2>', methods=['GET'])
def comparar_shap(id_escola_1, id_escola_2):
    """
    Endpoint para comparar os SHAP values de duas escolas.
    """
    # Pega os dados e calcula o SHAP para a primeira escola.
    escola_1_data = df_master[df_master['ID_ESCOLA'] == id_escola_1][FEATURES]
    if escola_1_data.empty: return jsonify({"error": f"Escola {id_escola_1} não encontrada"}), 404
    shap_values_1 = explainer.shap_values(escola_1_data)[0]
    
    # Pega os dados e calcula o SHAP para a segunda escola.
    escola_2_data = df_master[df_master['ID_ESCOLA'] == id_escola_2][FEATURES]
    if escola_2_data.empty: return jsonify({"error": f"Escola {id_escola_2} não encontrada"}), 404
    shap_values_2 = explainer.shap_values(escola_2_data)[0]
    
    # Calcula a diferença e formata para a tabela.
    tabela_comparativa = []
    for i, feature in enumerate(FEATURES):
        tabela_comparativa.append({
            "feature": feature,
            "shap_escola_1": float(shap_values_1[i]),
            "shap_escola_2": float(shap_values_2[i]),
            "diferenca": float(shap_values_1[i] - shap_values_2[i])
        })
    
    return jsonify(tabela_comparativa)

# Endpoint para o contrafactual (a ser implementado no frontend depois).
@app.route('/api/contrafactual', methods=['POST'])
def simular_contrafactual():
    """
    Recebe novos valores para as features e retorna a nova média predita.
    """
    dados_simulacao = request.json
    escola_id = dados_simulacao['escola_id']
    novos_valores = dados_simulacao['features']

    # Pega o vetor de features original da escola.
    vetor_original = df_master[df_master['ID_ESCOLA'] == escola_id][FEATURES].iloc[0].copy()

    # Atualiza o vetor com os valores da simulação.
    for feature, value in novos_valores.items():
        if feature in vetor_original:
            vetor_original[feature] = value

    # Faz a predição com os dados modificados.
    predicao = model.predict(vetor_original.to_frame().T)
    
    return jsonify({'media_simulada': float(predicao[0])})


# --- 6. Execução do Servidor ---
if __name__ == '__main__':
    # Roda o servidor na porta 5000 e permite depuração.
    print("Servidor Flask rodando em http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
