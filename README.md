# Dashboard Educacional

Este projeto utiliza Flask, XGBoost e SHAP para fornecer análises sobre a performance e similaridade de escolas, permitindo comparações baseadas em custo-benefício (distância vs. crescimento de performance predita).

## Pré-requisitos

Para executar o projeto num ambiente de desenvolvimento isolado, é necessário instalar as seguintes ferramentas na sua máquina:

### 1. Podman
O Podman é utilizado como motor de contentores em alternativa ao Docker.
- **Linux**: Utilize o gestor de pacotes da sua distribuição (ex: `sudo pacman -S podman` no Arch Linux, ou `sudo apt install podman`).
- **macOS / Windows**: Siga as instruções no [site oficial do Podman](https://podman.io/docs/installation).
  - *Nota para macOS/Windows*: Após a instalação, inicialize a máquina do Podman executando `podman machine init` e `podman machine start`.

### 2. VS Code e Extensões
- Instale o [Visual Studio Code](https://code.visualstudio.com/).
- Instale a extensão **Dev Containers** (`ms-vscode-remote.remote-containers`).

### 3. uv (Gestor de Pacotes Python)
O `uv` é utilizado para gerir dependências e ambientes virtuais de forma extremamente rápida. Instale executando no seu terminal:
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

```

---

## Configuração do Ambiente (Devcontainer)

Recomendamos a utilização do Devcontainer incluído no repositório para garantir que o ambiente de execução é idêntico para todos.

### Passo 1: Configurar o VS Code para usar o Podman

1. No VS Code, abra as configurações (`Ctrl` + `,` ou `Cmd` + `,`).
2. Pesquise por `Dev Containers: Docker Path`.
3. Altere o valor de `docker` para `podman`.

### Passo 2: Iniciar o Devcontainer

1. Abra a pasta raiz deste projeto no VS Code.
2. Prima `F1`, digite **Dev Containers: Reopen in Container** e prima Enter.
3. O VS Code irá construir a imagem a partir do `Dockerfile` e iniciar o contentor.
4. O processo executará automaticamente `uv sync` no final para criar o ambiente virtual (em `.venv`) e instalar todas as dependências especificadas no `uv.lock`.

---

## Executar a Aplicação

Com o Devcontainer em execução, o ambiente Python já estará configurado corretamente.

### 1. Sincronizar Dependências (Apenas se necessário)

Caso tenha feito alterações no ficheiro `pyproject.toml` ou se o ambiente não sincronizou automaticamente, execute no terminal do VS Code:

```bash
uv sync

```

### 2. Iniciar o Servidor Flask

Para arrancar com a API de backend:

```bash
cd page
uv run python app.py

```

O servidor Flask ficará ativo em `http://127.0.0.1:5000`.
As portas estarão automaticamente mapeadas para a sua máquina local, permitindo-lhe aceder ao dashboard e aos *endpoints* da API através do seu navegador habitual.

## Notas Adicionais sobre a Arquitetura

* A ordenação das escolas similares no *endpoint* do dashboard tem em conta o cálculo dinâmico de custo-benefício, que divide a distância percentual pela expectativa de crescimento da performance baseada nas previsões do XGBoost. As previsões base do modelo são pré-calculadas no arranque do servidor (`app.py`) para otimizar o tempo de resposta.