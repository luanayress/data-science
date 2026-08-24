# Churn Model v4 — expansão leakage-safe

## Objetivo

O v4 amplia o challenger v3 sem alterar o champion operacional v2. O experimento usa o mesmo split estratificado
80/20 (`random_state=42`) e seleciona features somente por validação cruzada no treino. O holdout permanece fora de
seleção, calibração e escolha de threshold.

## Contrato bancário

O melhor grupo foi `plus_demographics`, com dez entradas: `CreditScore`, `Age`, `Tenure`, `Balance`,
`NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Geography` e `Gender`.

`RowNumber`, `CustomerId`, `Surname`, `Exited`, `Complain` e `Complain_With_Low_Satisfaction` foram explicitamente
excluídos. O grupo de 13 features, que adicionava satisfação, tipo de cartão e pontos, teve PR-AUC de CV inferior e
não foi escolhido.

O pipeline persiste imputação, padronização numérica, one-hot encoding com categorias desconhecidas,
HistGradientBoosting e calibração sigmoid em um único artefato.

## Resultados

| Métrica holdout | v3 | v4 |
|---|---:|---:|
| PR-AUC | 0,6199 | 0,7310 |
| ROC-AUC | 0,8297 | 0,8783 |
| Precision | 0,5872 | 0,6312 |
| Recall | 0,5613 | 0,6544 |
| F1 | 0,5739 | 0,6426 |
| Brier | 0,1161 | 0,0972 |
| Log loss | 0,3738 | 0,3213 |

O threshold balanceado é `0,33`. A recomendação é técnica: o v4 ainda precisa de validação financeira real e
observação em shadow antes de promoção.

## Readiness gates

Os gates foram executados com resultado `NOT_READY_FOR_CANARY`:

| Gate | Resultado | Evidência |
|---|---|---|
| Bootstrap pareado | PASS | PR-AUC Δ IC95% `[0,0838; 0,1385]`; F1 Δ `[0,0375; 0,0990]` |
| Calibração | PASS | sigmoid venceu por Brier OOF `0,101954` |
| Importância | PASS | Age e NumOfProducts dominam; relatório de permutação persistido |
| Contrato v4 | PASS | endpoints `/v4/predict` e `/v4/predict-batch` exigem as dez features |
| Fairness | FAIL | Geography: recall gap `19,6 p.p.` e FPR gap `12,5 p.p.`; limite `10 p.p.` |
| Valor financeiro | BLOCKED | valores reais não configurados |
| Capacidade | BLOCKED | limite de campanha não configurado |
| SLO de latência | BLOCKED | SLO não configurado |

O benchmark local mediu p95 de aproximadamente `69 ms` para uma linha e `261 ms` para um batch de 2.000 linhas.
Esses números são evidência local, não um SLO de produção.

O canary não foi iniciado. Para desbloqueá-lo é necessário tratar a disparidade geográfica e preencher custos,
capacidade e SLO sem alterar retroativamente os limites para acomodar o resultado observado.

## Operação

Treinar novamente:

```powershell
python scripts/train_feature_expansion.py
```

Shadow seguro:

```powershell
$env:MODEL_VERSION="v2"
$env:SHADOW_MODEL_VERSION="v4"
python -m uvicorn app.api:app --reload --port 8000
```

O shadow v4 somente compara requisições com as dez features completas. Payloads legados são registrados como
`skipped` e nunca alteram a resposta v2.

Ativação explícita, somente após aprovação:

```powershell
$env:MODEL_VERSION="v4"
```

Rollback:

```powershell
$env:MODEL_VERSION="v2"
Remove-Item Env:SHADOW_MODEL_VERSION -ErrorAction SilentlyContinue
```
