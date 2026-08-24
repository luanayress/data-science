# Arquitetura após a Fase 1

## Resultado

A Fase 1 substituiu o caminho de inferência v1 fragmentado por um pipeline sklearn v2 único, treinado a partir de dados brutos. O snapshot histórico permanece em `architecture-current.md`.

```text
data/raw/Customer-Churn-Records.csv
  -> split antes de qualquer fit
  -> sklearn Pipeline
       -> ChurnFeatureEngineer
       -> StandardScaler
       -> GradientBoostingClassifier
  -> models/v2/pipeline.joblib + metadata.json
  -> InferencePipeline
  -> FastAPI
  -> Streamlit
```

## Decisões

- A estrutura atual de `src/features`, `src/models` e `src/pipelines` foi mantida para reduzir risco.
- `feature_contract.py` é a fonte oficial de target e features.
- O pipeline aceita dados raw e encapsula todo estado ajustado.
- A versão v1 foi preservada para rollback; a API usa v2 por padrão e aceita override por `MODEL_VERSION`.
- Os contratos `/health`, `/model-info`, `/predict` e `/predict-batch` foram preservados.

## Componentes ativos

| Componente | Responsabilidade |
|---|---|
| `src/features/feature_contract.py` | Contrato único. |
| `src/features/build_features.py` | Transformer determinístico. |
| `src/pipelines/training_pipeline.py` | Split, fit, avaliação e persistência. |
| `src/models/registry.py` | Save/load de pipeline e metadata versionados. |
| `src/pipelines/inference_pipeline.py` | Predição sem fit. |
| `scripts/train_churn.py` | Entry point fino. |
| `app/api.py` | API compatível usando v2. |

## Riscos remanescentes

- Apenas três dos 15 campos HTTP alimentam o modelo; isso é explícito, mas limita poder preditivo.
- A API e o frontend continuam monolíticos.
- Analytics ainda é simulado.
- Artefatos e scripts legados permanecem por rollback/compatibilidade.
- Threshold 0,5 ainda não foi otimizado por objetivo de negócio.
- Eventos FastAPI `on_event` geram aviso de depreciação e devem migrar para lifespan em fase estrutural.
