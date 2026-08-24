# Treinamento e inferência oficiais

## Contrato

- Dataset oficial: `data/raw/Customer-Churn-Records.csv`
- Alvo: `Exited`
- Entradas do modelo: `Age`, `Tenure`, `NumOfProducts`
- Features internas: `NumOfProducts`, `Age_Squared`, `Age_Tenure_Interaction`
- Fonte de verdade: `src/features/feature_contract.py`

O contrato HTTP continua recebendo 15 campos. Os outros 12 campos são preservados por compatibilidade, mas estão declarados em `IGNORED_HTTP_FEATURES` e não afetam a v2.

## Treinamento

```text
Customer-Churn-Records.csv (raw)
        -> validação de RAW_FEATURES + Exited
        -> train_test_split estratificado
        -> pipeline.fit(X_train, y_train)
             -> ChurnFeatureEngineer (stateless)
             -> StandardScaler (fit somente no treino)
             -> GradientBoostingClassifier
        -> avaliação em X_test
        -> models/v2/pipeline.joblib
        -> models/v2/metadata.json
```

Entry point oficial:

```powershell
python scripts/train_churn.py
```

`train_and_save.py` permanece apenas como wrapper compatível e está deprecado. A v1 não é sobrescrita.

## Inferência

```text
Payload raw -> DataFrame -> pipeline.predict_proba() -> threshold -> resposta
```

`InferencePipeline` carrega o pipeline completo uma vez no startup. Nenhuma chamada a `fit` ou `fit_transform` ocorre em inferência. O mesmo objeto atende previsões individuais e batches, portanto o resultado de um cliente é independente dos demais itens do lote.

## Persistência e metadata

`models/v2/metadata.json` registra versão, algoritmo, alvo, features raw/internas, campos HTTP ignorados, data, tamanhos de treino/teste, seed, thresholds e métricas de teste.

## Classificação dos fluxos

| Caminho | Estado |
|---|---|
| `scripts/train_churn.py` | ACTIVE — entrypoint oficial v2. |
| `src/pipelines/training_pipeline.py` | ACTIVE — implementação oficial. |
| `src/pipelines/inference_pipeline.py` | ACTIVE — inferência oficial. |
| `train_and_save.py` | DEPRECATED — wrapper para o entrypoint oficial. |
| `save_models.py` | LEGACY/DEPRECATED — layout v1 antigo. |
| `scripts/regenerate_models.py` | LEGACY/DEPRECATED — layout v1 antigo. |
| `model_deployment.py` | LEGACY/DEPRECATED — loader direto antigo. |
| `app/model_loader.py` | LEGACY/DEPRECATED — loader direto antigo. |
| `scripts/migrate_pickles_to_joblib.py` | MIGRATION — somente migração histórica. |
| `scripts/reorganize_models.py` | MIGRATION — somente reorganização histórica. |
| notebooks | LEGACY/EXPLORATORY — não são o pipeline de produção. |

## Validação

Os testes cobrem transformação de Age/Tenure, ausência de fit em inferência, invariância de batch, equivalência após serialização e rotas FastAPI reais.
