# Promoção controlada do modelo de churn

## Estado

- Champion operacional: v2, GradientBoosting, threshold `0.50`.
- Challenger: v3, HistGradientBoosting calibrado com sigmoid.
- Status: `PROMOTION_RECOMMENDED`.
- Classificação: `PROMOTE_V3_TECHNICALLY`.

A v3 ainda não foi promovida. Faltam valores financeiros reais e confirmação da capacidade operacional da campanha.

## Avaliação

Execute sem retreinar:

```powershell
python scripts/evaluate_model_promotion.py
```

O comando carrega os artefatos existentes e os resultados OOF da Fase 2B. Os arquivos em `reports/promotion/` documentam cenários, sensibilidade de custos, divergências e decisão.

O modo padrão é `RELATIVE_COST`, configurado em `configs/churn_business.yaml`. `false_negative_cost` e `false_positive_cost` representam unidades relativas, não valores monetários. O modo `FINANCIAL_VALUE` requer explicitamente valor médio do cliente, custo da campanha, taxa de sucesso e custo de contato falso positivo.

## Thresholds

Thresholds pertencem ao metadata de cada modelo:

- v2: `0.50`;
- v3 balanced: `0.34`;
- v3 high recall: `0.21`;
- v3 high precision: `0.54`.

Configure opcionalmente:

```powershell
$env:CHURN_THRESHOLD_PROFILE="balanced"
```

Sem perfil, `selected_threshold` do metadata é utilizado. Os valores da v3 foram escolhidos com probabilidades OOF, nunca com o holdout.

## Shadow mode

Para executar v3 paralelamente sem mudar a resposta oficial:

```powershell
$env:MODEL_VERSION="v2"
$env:SHADOW_MODEL_VERSION="v3"
python -m uvicorn app.api:app --port 8000
```

Os dois artefatos são carregados uma vez no lifespan. A resposta HTTP continua vindo exclusivamente da v2. Logs incluem apenas request ID, versões, probabilidades, delta e decisões, sem payload completo.

## Promoção

Uma promoção posterior e explícita pode ser validada com:

```powershell
$env:MODEL_VERSION="v3"
$env:CHURN_THRESHOLD_PROFILE="balanced"
```

Antes de tornar isso default, informar capacidade máxima de contatos ou campaign rate e parâmetros financeiros reais. Só então a decisão pode evoluir para `PROMOTE_V3_BUSINESS_VALIDATED`.

## Rollback

Rollback não exige alterar artefatos:

```powershell
$env:MODEL_VERSION="v2"
Remove-Item Env:SHADOW_MODEL_VERSION -ErrorAction SilentlyContinue
Remove-Item Env:CHURN_THRESHOLD_PROFILE -ErrorAction SilentlyContinue
```

Reinicie a API e confirme `/health`, `/model-info`, `/predict` e `/predict-batch`.
