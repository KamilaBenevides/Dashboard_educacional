
# Sistema de Apoio à Decisão Educacional com XAI (SAD-Educacional)

Este projeto é fruto de uma dissertação de mestrado e consiste em um **Sistema de Apoio à Decisão (SAD)** para gestão educacional. A ferramenta utiliza **Machine Learning (XGBoost)** e **Inteligência Artificial Explicável (SHAP)** para analisar o desempenho de escolas públicas brasileiras com base em dados contextuais do SAEB, permitindo comparações justas (benchmarking contextual) e simulações de cenários.

## 📋 Funcionalidades

-   **Previsão de Desempenho:** Estima a nota média da escola com base em variáveis socioeconômicas.
    
-   **Explicabilidade (XAI):** Utiliza SHAP Values para mostrar exatamente quais fatores (ex: infraestrutura, escolaridade dos pais) aumentam ou diminuem a nota da escola.
    
-   **Vizinhança Contextual:** Algoritmo próprio que identifica escolas com perfis socioeconômicos similares, mas desempenhos diferentes, para fins de comparação.
    
-   **Simulador Contrafactual:** Permite que o gestor altere variáveis (ex: "E se 100% dos alunos tivessem internet?") para ver o impacto projetado na nota.
    

## 🛠️ Tecnologias Utilizadas

-   **Backend:** Python 3.10+, Flask.
    
-   **Machine Learning:** XGBoost, SHAP, Scikit-learn, Pandas, NumPy.
    
-   **Frontend:** HTML5, JavaScript (Vanilla + jQuery), Chart.js, TailwindCSS.
    

## 🚀 Como Rodar o Projeto

Siga os passos abaixo para executar o projeto em sua máquina local.

### 1. Pré-requisitos

Certifique-se de ter o **Python** instalado (recomendado versão 3.10 ou superior devido à compatibilidade dos arquivos `.pkl`).

Verifique sua versão com:

```
python --version

```

### 2. Clonar o Repositório

```
git clone [https://github.com/KamilaBenevides/Dashboard_educacional.git](https://github.com/KamilaBenevides/Dashboard_educacional.git)
cd NOME_DO_REPO

```

### 3. Configurar o Ambiente Virtual (Recomendado)

É uma boa prática criar um ambiente virtual para isolar as dependências do projeto.

No Windows:

```
python -m venv venv
venv\Scripts\activate

```

No Linux/Mac:

```
python3 -m venv venv
source venv/bin/activate

```

### 4. Instalar Dependências

Instale as dependências listadas no arquivo de requisitos:

```
pip install -r page/requirements.txt

```

_(Nota: Ajuste o caminho do requirements.txt se ele estiver na raiz)_

### 5. Organização dos Arquivos de Dados

⚠️ **Importante:** A estrutura de pastas é crítica para o funcionamento do sistema, pois o código busca os modelos na pasta anterior (`../`). Certifique-se de que os arquivos `.pkl` e `.csv` estejam na raiz e o código na pasta `page`.

A estrutura deve ser organizada da seguinte forma:

```
/projeto (Raiz do Repositório)
├── modelo_xgboost.pkl              # Modelo treinado (Pasta anterior à page)
├── shap_explainer.pkl              # Explainer do SHAP (Pasta anterior à page)
├── dataset_reduzido_renomeadas2.csv
├── escolas_com_cep.csv             # (Verifique o caminho ../../ no código se necessário)
└── page                            # Pasta da Aplicação
    ├── app.py                      # Servidor Flask
    ├── index.html                  # Interface Frontend
    ├── similarity_calculator.py
    └── requirements.txt

```

### 6. Executar a Aplicação

Como o código da aplicação está dentro da pasta `page`, você deve entrar nela antes de executar o servidor:

```
cd page
python app.py

```

Você verá uma mensagem indicando que o servidor está rodando (geralmente em `http://127.0.0.1:5000`).

### 7. Acessar o Dashboard

Com o servidor rodando, abra o arquivo `index.html` diretamente no seu navegador ou acesse o endereço indicado no terminal.

## 📊 Estrutura do Código

-   `page/app.py`: API principal que carrega o modelo (da pasta pai), calcula os valores SHAP e serve os dados.
    
-   `page/similarity_calculator.py`: Classe responsável pela lógica de distância de percentil.
    
-   `page/index.html`: Dashboard interativo que consome a API.
    

