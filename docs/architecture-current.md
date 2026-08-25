# Arquitetura atual — diagnóstico da Fase 0

> Snapshot analisado em 2026-08-24. Este documento descreve o repositório como ele está; não propõe que o estado atual seja o estado desejado. A Fase 0 não moveu nem refatorou código.

## Estado atual

O projeto implementa previsão de churn com três camadas principais:

1. um frontend Streamlit monolítico em `app/app.py`;
2. uma API FastAPI monolítica em `app/api.py`;
3. módulos de dados, features, modelos, pipelines, monitoramento e utilidades em `src/`.

O caminho ativo de produção usa um `GradientBoostingClassifier`, um `StandardScaler` e metadados sob `models/v1/`. O modelo ativo espera três features:

- `NumOfProducts`;
- `Age_Squared_StandardScaled`;
- `Age_Tenure_Interaction_MinMaxScaled`.

Há 10.000 registros no dataset bruto. Os datasets presentes são:

| Dataset | Dimensão | Papel atual |
|---|---:|---|
| `data/raw/Customer-Churn-Records.csv` | 10.000 × 18 | Fonte bruta de churn bancário; alvo `Exited`. |
| `data/processed/Customer_Churn_Engineered_Features.csv` | 10.000 × 61 | Features criadas pelo notebook de engenharia. |
| `data/processed/Customer_Churn_Final_Features.csv` | 10.000 × 6 | Seleção final; contém cinco features e `Exited`. |

O ambiente inspecionado usa Python 3.8.10, FastAPI 0.124.4, Pydantic 2.10.6, Streamlit 1.40.1, scikit-learn 1.3.2, pandas 2.0.3 e NumPy 1.24.4.

## Arquitetura

```text
Usuário
  |
  v
app/app.py (Streamlit :8501)
  |  requests HTTP
  v
app/api.py (FastAPI :8000)
  |-- app/schema.py (contratos HTTP/Pydantic)
  |-- src/monitoring/monitor.py (drift)
  `-- src/pipelines/inference_pipeline.py
        |-- src/features/build_features.py
        `-- src/models/registry.py
              `-- models/v1/{model,scaler}/

Treinamento
  |-- notebooks/01_eda.ipynb
  |-- notebooks/02_feature_engineering.ipynb
  |-- notebooks/03_modeling.ipynb
  |-- train_and_save.py (caminho executável que gera os artefatos ativos)
  `-- src/pipelines/training_pipeline.py (pipeline alternativo, atualmente inconsistente)
```

### Mapa de responsabilidades

| Área | Arquivos principais | Responsabilidade observada |
|---|---|---|
| Frontend | `app/app.py` | UI, navegação, chamadas HTTP, tratamento de erro, métricas, gráficos, download e analytics simulado. |
| Backend | `app/api.py` | Criação da aplicação, CORS, ciclo de vida, middleware, rotas, conversão para DataFrame, threshold, logging e tratamento de erro. |
| Schemas | `app/schema.py` | Requests e responses Pydantic. |
| Inferência | `src/pipelines/inference_pipeline.py` | Carrega artefatos, cria features, aplica scaler e prediz. |
| Treinamento | `train_and_save.py`, `src/pipelines/training_pipeline.py`, notebooks e scripts auxiliares | Existem vários caminhos concorrentes para treinar, salvar e migrar artefatos. |
| Features | `src/features/build_features.py`, `feature_store.py`, `feature_contract.py`, `transformers.py` | Implementações sobrepostas de engenharia, validação e transformação. |
| Modelos | `src/models/train.py`, `evaluate.py`, `predict.py`, `registry.py` | Treino, avaliação, predição genérica e persistência. |
| Dados | `src/data/load_data.py`, `split.py`, `validation.py` | Leitura, escrita, validação e split. |
| Monitoramento | `src/monitoring/monitor.py` | KS para features numéricas e qui-quadrado para categóricas. |
| Configuração | `configs/*.yaml`, `src/utils/config.py`, `src/utils/paths.py` | YAML com override `CFG__...`; somente parte da configuração é realmente consumida. |
| Testes | `tests/` e scripts de smoke test | Testes unitários de schemas, features, utilitários de modelo e drift; cobertura de integração limitada. |
| Notebooks | `notebooks/01_eda.ipynb`, `02_feature_engineering.ipynb`, `03_modeling.ipynb` | EDA, geração de features, seleção e comparação inicial de modelos. |

## Fluxo de treinamento

### Caminho que atualmente gera os artefatos ativos

```text
Customer_Churn_Final_Features.csv
  -> seleciona 3 features e Exited
  -> split estratificado 80/20 (random_state=42)
  -> StandardScaler ajustado somente em X_train
  -> GradientBoostingClassifier
  -> avaliação por accuracy em treino e teste
  -> models/v1/model/model.joblib
  -> models/v1/scaler/scaler.joblib
  -> models/v1/model/metadata.json
```

Esse fluxo está em `train_and_save.py`. Na última execução registrada, obteve accuracy de treino `0,8539` e de teste `0,8325`.

### Origem das features processadas

`notebooks/02_feature_engineering.ipynb` lê o CSV bruto, cria interações, polinômios, bins e encodings, ajusta `StandardScaler` e `MinMaxScaler`, faz seleção e grava os CSVs processados. O notebook ajusta os scalers antes do split de modelagem, usando o dataset completo; portanto, as colunas escaladas incorporam estatísticas do futuro conjunto de teste (vazamento de dados).

`notebooks/03_modeling.ipynb` remove `Complain` e `Complain_With_Low_Satisfaction` por leakage e compara Logistic Regression, Random Forest e Gradient Boosting. O modelo implantado conserva as três features restantes.

### Pipeline alternativo em `src/`

`run_training_pipeline()` pretende executar carga, validação, engenharia, split, scaling, treino, avaliação e persistência. No estado atual ele não é executável de ponta a ponta:

- `build_features()` retorna somente três features e descarta o alvo;
- `get_features_for_modeling()` procura por padrão o alvo `Churn`;
- o dataset real usa `Exited`;
- o parâmetro `config_file` não é usado;
- `training.yaml` é aninhado, enquanto o código procura `test_size` e `model_params` no nível raiz, fazendo os valores configurados serem ignorados.

### Outros caminhos concorrentes

- `save_models.py` persiste artefatos no layout legado da raiz de `models/`.
- `scripts/regenerate_models.py` grava `models/v1/model.pkl` e `scaler.pkl`, layout diferente do registro ativo.
- `scripts/migrate_pickles_to_joblib.py` e `scripts/reorganize_models.py` migram/reorganizam layouts antigos.
- `model_deployment.py` e `app/model_loader.py` carregam o layout legado e duplicam parte da inferência.

## Fluxo de inferência

```text
Payload Pydantic
  -> DataFrame
  -> build_features()
       Age^2
       Age * Tenure
       ajuste de StandardScaler e MinMaxScaler na própria requisição
  -> seleção das 3 features descritas nos metadados
  -> scaler persistido em models/v1/scaler/scaler.joblib
  -> GradientBoostingClassifier.predict_proba()
  -> threshold 0,5
  -> prediction, probability e confidence
```

No startup da API, `InferencePipeline(version="v1")` carrega uma única vez:

- `models/v1/model/model.joblib`;
- `models/v1/scaler/scaler.joblib`;
- `models/v1/model/metadata.json`.

Isso evita reload por request. Entretanto, existe um defeito crítico no preprocessing online: `build_features()` ajusta scalers novos sobre o próprio request. Em previsão individual, uma única observação sempre vira zero nas duas features derivadas antes da aplicação do scaler persistido. Uma verificação com `(Age=30, Tenure=5)` e `(Age=60, Tenure=40)`, mantendo `NumOfProducts=2`, produziu vetores finais idênticos. Em batch, o resultado de um cliente depende da composição do lote. Assim, idade e tenure não influenciam corretamente a predição online, apesar de serem apresentadas como inputs.

Há ainda dupla transformação: o CSV final já contém features escaladas, `train_and_save.py` ajusta outro `StandardScaler`, e a inferência recria as features com scalers ajustados no request antes de aplicar esse segundo scaler.

## Fluxo frontend → API

```text
app/app.py
  -> resolve API_URL (ambiente, Streamlit secrets ou localhost:8000)
  -> GET /health antes de renderizar as páginas
  -> GET /model-info para sidebar/página de modelo
  -> POST /predict para formulário individual
  -> POST /predict-batch para CSV
  -> renderiza a resposta e mensagens diretamente
```

As chamadas `requests`, timeouts, mensagens de erro e interpretação de status estão dentro do mesmo arquivo das páginas. Não existe cliente HTTP dedicado. `check_api_health()` usa `except:` amplo; as demais chamadas capturam `Exception` e escrevem diretamente na UI.

O frontend envia 15 campos. O pipeline usa apenas `Age`, `Tenure` e `NumOfProducts`; os outros 12 campos são validados pelo schema, mas ignorados pelo modelo atual.

## Endpoints

| Método | Rota | Entrada | Saída/efeito | Dependências |
|---|---|---|---|---|
| GET | `/` | — | Índice simples e lista de rotas. | Nenhuma. |
| GET | `/health` | — | Status, modelo carregado, versão e timestamp. | `app.state.pipeline`. |
| GET | `/model-info` | — | Tipo, versão, data, métricas e features. | Metadados do `InferencePipeline`. |
| POST | `/predict` | `PredictionRequest` | Uma `PredictionResponse`. | Pipeline, threshold hardcoded `0.5`. |
| POST | `/predict-batch` | `BatchPredictionRequest` | Lista de predictions. | Pipeline, threshold hardcoded `0.5`. |
| POST | `/monitor/report` | Dois CSVs e `alpha` | Relatório de drift. | `ModelMonitor`, arquivos temporários. |

Todas as rotas estão em `app/api.py`; não há routers separados. O middleware adiciona `X-Request-ID`. CORS usa `ALLOWED_ORIGINS` ou quatro origens locais hardcoded.

### Schemas Pydantic

- `PredictionRequest`: 15 campos, com validação numérica parcial; enums `ContractType` e `InternetServiceType` existem, mas os campos correspondentes são tipados como `str`.
- `PredictionResponse`: classe, probabilidade e confiança.
- `BatchPredictionRequest`/`BatchPredictionResponse`: envelopes de batch.
- `ModelInfo`: metadados e métricas opcionais.
- `HealthCheck`: estado da API.

## Modelos existentes

### Ativo

| Componente | Local | Observação |
|---|---|---|
| Gradient Boosting | `models/v1/model/model.joblib` | Carregado pelo `ModelRegistry`. |
| StandardScaler | `models/v1/scaler/scaler.joblib` | Carregado pelo `ModelRegistry`. |
| Metadados | `models/v1/model/metadata.json` | Declara três features, train/test score e tipo. |

### Legados ou redundantes

- `models/gradient_boosting_model.pkl`;
- `models/scaler_standard.pkl` e `models/scaler_minmax.pkl`;
- `models/preprocessing_config.pkl`;
- cópias equivalentes diretamente em `models/v1/`;
- `models/v1/preprocessor/preprocessor.joblib`.

Não há manifesto único que declare quais arquivos são ativos, sua compatibilidade, hash do dataset ou versão do código. O diretório mistura o layout atual e pelo menos dois layouts legados.

## Problemas encontrados

### Críticos

1. **Inconsistência treino/inferência:** scalers de features são reajustados por request. Previsões individuais eliminam o efeito de `Age` e `Tenure`; batches são dependentes da composição do lote.
2. **Pipeline oficial alternativo quebrado:** `src/pipelines/training_pipeline.py` perde o alvo e procura `Churn`, enquanto o dataset usa `Exited`.
3. **Vazamento na criação do dataset:** as features escaladas do notebook usam estatísticas do dataset completo antes do split.
4. **Configuração de inferência divergente:** `configs/inference.yaml` aponta para `model.pkl`, `scaler.pkl` e metadata na raiz de `models/v1`, mas o runtime usa subdiretórios com `.joblib` e não lê essa configuração.
5. **Imagem Docker não é reproduzível:** o `Dockerfile` executa Uvicorn, porém `requirements.txt` não declara `fastapi` nem `uvicorn`. Também faltam dependências importadas pelo código, como `requests`, `plotly` e `PyYAML`.

### Altos

1. **Múltiplas fontes de verdade:** features aparecem em notebook, `train_and_save.py`, `build_features.py`, `feature_store.py`, YAML e metadados, com conjuntos diferentes.
2. **Contrato divergente:** `required_raw_features()` exige `HasCrCard`, `build_features()` não exige esse campo e `PredictionRequest` não o contém.
3. **Features declaradas versus usadas:** `features.yaml` declara cinco features; o modelo ativo usa três.
4. **Campos de API ignorados:** 12 dos 15 inputs do frontend não entram no modelo.
5. **Métricas incompletas:** o artefato recente guarda accuracy de treino/teste, mas não F1, ROC-AUC, precision, recall ou data de treinamento; `/model-info` pode retornar campos vazios.
6. **Analytics simulado:** `show_analytics()` gera 100 registros aleatórios e pode ser confundido com resultado real.
7. **Testes não cobrem endpoints reais:** os 21 testes cobrem principalmente schemas e funções. Não há teste pytest do ciclo de vida e das rotas; `test_api_endpoints.py` é um script manual desatualizado e sem assertions.

### Médios

1. `app/app.py` mistura UI, cliente HTTP, regras de apresentação, analytics e estado.
2. `app/api.py` mistura bootstrap, rotas, dependências, monitoramento e configuração.
3. Threshold `0.5`, limite de alta confiança `0.8`, versão `v1`, portas, URLs e timeouts estão espalhados/hardcoded.
4. `src/models/predict.py` e `InferencePipeline` duplicam lógica de probabilidade, threshold e confiança com semânticas ligeiramente diferentes.
5. `FeatureStore` duplica `build_features()` e aparentemente não participa do caminho ativo.
6. `ModelDeployment` e `app/model_loader.py` duplicam carregamento/preprocessing usando artefatos legados.
7. Os notebooks usam caminhos relativos como `Customer-Churn-Records.csv`, diferentes da estrutura real `data/raw/`, e dependem do diretório de execução.
8. `run_dashboard.bat` aponta para `app.py` na raiz e valida artefatos legados, enquanto o frontend está em `app/app.py` e o backend usa `models/v1/...`.
9. O `Dockerfile` define `DATA_DIR`, `CONFIG_DIR` e `LOG_DIR`, mas os helpers de paths só respeitam `MODEL_DIR`.
10. Documentos existentes contêm comandos e layouts obsoletos ou afirmações mais fortes que os testes sustentam.
11. Há texto com mojibake em mensagens e documentação, prejudicando legibilidade e apresentação do portfólio.

## Débitos técnicos

- ausência de um pipeline sklearn único que persista engenharia + preprocessing + estimador;
- ausência de versionamento semântico e validação de compatibilidade dos artefatos;
- ausência de identificação/versionamento do dataset de treino;
- ausência de separação entre rotas, serviços e dependências da API;
- ausência de um cliente HTTP reutilizável no frontend;
- ausência de serviço de analytics real e persistência de batches;
- ausência de testes de integração para startup, health, model-info, predict, batch e indisponibilidade do modelo;
- ausência de teste de paridade treino/inferência e invariância entre predição individual e batch;
- requirements incompletos e sem versões travadas;
- scripts históricos executáveis convivendo com o fluxo atual sem marcação explícita de depreciação;
- tratamento amplo de exceções, que reduz observabilidade e pode transformar defeitos em respostas 400;
- teste de `ModelRegistry` não exercita save/load e cria `models/test` em vez de isolar `MODEL_DIR`;
- testes de drift usam aleatoriedade sem seed, com risco de flutuação.

## Riscos de quebrar o comportamento atual

1. Alterar nomes ou ordem das três features quebra scaler, modelo e metadados já persistidos.
2. Corrigir o preprocessing muda materialmente as probabilidades atuais; exige retreino e testes de regressão, não apenas troca no runtime.
3. Remover artefatos legados pode quebrar scripts antigos ainda documentados.
4. Trocar schemas pode quebrar o formulário Streamlit, CSVs de batch e consumidores externos.
5. Mover `app.py`/`api.py` sem compatibilidade quebra comandos do README, imports e Docker CMD.
6. Centralizar configuração sem mapear precedência entre ambiente, secrets e YAML pode mudar endpoints, CORS e diretórios.
7. Substituir analytics simulado exige definir uma fonte real e seu ciclo de vida; a API atualmente não persiste previsões.

## Recomendações

### P0 — estabilização imediata

1. Definir um único contrato de dados, alvo e features, compartilhado por treino e inferência.
2. Retreinar e persistir um pipeline completo com transformers ajustados apenas no treino; validar paridade individual/batch.
3. Escolher um único entrypoint de treinamento e marcar scripts/notebooks históricos como legados.
4. Adicionar testes de integração de todas as rotas e testes específicos para o defeito de preprocessing identificado.
5. Corrigir `requirements.txt` e validar build/execução em ambiente limpo.
6. Consolidar a configuração com precedência documentada e remover referências a layouts inexistentes.

### Próxima reorganização arquitetural

1. Separar rotas FastAPI, dependências, schemas e serviços sem mudar inicialmente os contratos HTTP.
2. Extrair cliente HTTP do Streamlit e manter as páginas focadas em renderização.
3. Criar um módulo de churn explícito e interfaces para futuros modelos, evitando diretórios vazios com implementações fictícias.
4. Isolar analytics simulado como fallback claramente rotulado e criar serviço para dados reais/batches.
5. Manter adaptadores temporários para comandos e imports antigos durante a migração.

## Evidências de validação da Fase 0

- Grafo existente do projeto consultado: 427 nós; relações principais de treino, inferência, API e testes confirmadas por inspeção direta.
- CSVs inspecionados quanto a dimensões e colunas.
- Artefatos ativos carregados por `InferencePipeline`.
- Teste direto confirmou que dois perfis distintos geram vetores individuais idênticos quando `NumOfProducts` é igual.
- TestClient confirmou respostas HTTP 200 para `/health`, `/model-info`, `/predict` e `/predict-batch` no snapshot analisado.
- Nenhuma refatoração ou movimentação de arquivo foi realizada nesta fase.
