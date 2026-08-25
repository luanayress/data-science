# Arquitetura após a Fase 2A

## Resultado

A Fase 2A separou transporte HTTP, orquestração e ML sem alterar o pipeline v2, suas features, thresholds ou artefatos. O entrypoint continua sendo `app/api.py` e o dashboard continua em `app/app.py`.

O diretório sugerido `app/api/routers/` não foi usado porque coexistir `app/api.py` e um pacote `app/api/` torna `import app.api` ambíguo em Python. Para preservar literalmente `uvicorn app.api:app`, os routers ficam em `app/routers/` e as dependências em `app/dependencies.py`.

## Antes e depois

Antes:

```text
Streamlit (UI + requests + demo analytics)
  -> FastAPI (bootstrap + lifecycle + rotas + lógica de aplicação)
     -> InferencePipeline / ModelMonitor
```

Depois:

```text
User
  |
  v
Streamlit app/app.py
  |
  v
ApiClient app/frontend/services/api_client.py
  |
  v
FastAPI app/api.py
  |-- Health Router
  |-- Model Info Router
  |-- Churn Router
  `-- Monitoring Router
        |
        v
     Services
        |
        v
InferencePipeline / ModelMonitor
        |
        v
ModelRegistry
        |
        v
models/v2/
```

## Fluxos

### Startup

`uvicorn` importa `app.api:app`; o lifespan resolve a configuração, carrega uma instância de `InferencePipeline(version)` e uma de `ModelMonitor` em `app.state`. O shutdown apenas registra o encerramento. Não há `on_event`.

### Request e dependency injection

O middleware cria `request.state.request_id` e devolve `X-Request-ID`. Dependências em `app/dependencies.py` leem os objetos já carregados de `request.app.state` e constroem services leves. Nenhum endpoint carrega modelo e nenhum service executa `fit`.

### Predição

`/predict` e `/predict-batch` validam os mesmos schemas, delegam ao `ChurnService` e este apenas converte schemas para DataFrame e chama `InferencePipeline.predict_with_confidence()`. Feature engineering, scaling e `predict_proba` continuam dentro do pipeline sklearn serializado.

### Monitoramento

O router lê os uploads e o `MonitoringService` coordena arquivos temporários, parsing CSV, chamada ao `ModelMonitor` e limpeza. Erros previsíveis são convertidos para o mesmo status HTTP 400; indisponibilidade permanece 503.

### Frontend

As páginas Streamlit chamam exclusivamente `ApiClient`. URL, timeouts, validação de status, parsing JSON e erros de conexão estão centralizados. Health usa cache de 15 segundos e model-info de 60 segundos; predições não usam cache.

Analytics usa `AnalyticsDataProvider`, atualmente com estado explícito `DEMO`. A UI exibe `DEMO DATA` e a interface admite futuras fontes `REAL` e `UNAVAILABLE`.

## Configuração

`app/core/config.py` resolve configuração nesta ordem:

```text
variável de ambiente -> configs/inference.yaml -> default seguro
```

Variáveis diretas: `MODEL_VERSION`, `MODEL_DIR`, `API_HOST`, `API_PORT`, `API_URL`, `ALLOWED_ORIGINS`, `PREDICTION_THRESHOLD`, `LOG_LEVEL`, `APP_ENV`, `API_TIMEOUT` e `API_BATCH_TIMEOUT`. O loader YAML existente e seus overrides `CFG__...` continuam reutilizados. O padrão de modelo é `v2` somente na configuração central; consumidores usam `Settings.model_version`.

## Compatibilidade

Permanecem estáveis os comandos `uvicorn app.api:app` e `streamlit run app/app.py`, os imports de `app.schema`, os cinco endpoints públicos, schemas, códigos esperados e `X-Request-ID`. Os arquivos em `models/v2/` não são modificados nesta fase.
