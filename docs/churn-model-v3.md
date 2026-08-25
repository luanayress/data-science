# Churn Model v3 — Fase 2B

## Problema e contrato

O experimento compara algoritmos sem alterar a pergunta de features. Todos recebem `Age`, `Tenure` e `NumOfProducts`; `ChurnFeatureEngineer` produz `NumOfProducts`, `Age_Squared` e `Age_Tenure_Interaction`, seguido pelo mesmo `StandardScaler`.

Dataset: `data/raw/Customer-Churn-Records.csv`, alvo `Exited`, 10.000 linhas. O split estratificado reproduz a v2: 8.000 linhas de treino e 2.000 de holdout, `random_state=42`. O holdout não participa de tuning, seleção, calibração ou thresholds.

## Metodologia

- `StratifiedKFold`, 5 folds, shuffle e seed 42.
- Refit por `average_precision`, a definição sklearn de PR-AUC usada neste projeto.
- Busca aleatória limitada: 8 combinações para Logistic Regression e 10 para RF, GB e HistGB.
- OOF probabilities geradas somente no treino para calibração e thresholds.
- Calibração sigmoid/isotonic avaliada com CV externa de 5 folds e CV interna de 3 folds.
- Holdout aberto uma única vez após escolher o challenger, a calibração e os thresholds.

XGBoost, LightGBM e CatBoost foram ignorados porque não estavam instalados. Nenhuma dependência opcional foi adicionada automaticamente.

## Resultados comparativos

| Modelo | CV ROC-AUC | CV PR-AUC | CV Recall | Holdout ROC-AUC | Holdout PR-AUC | Holdout F1 |
|---|---:|---:|---:|---:|---:|---:|
| DummyPrior | 0.5000 | 0.2037 | 0.0000 | 0.5000 | 0.2040 | 0.0000 |
| LogisticRegression | 0.7270 | 0.3587 | 0.6000 | 0.7474 | 0.3736 | 0.4995 |
| ProductionV2 | 0.8046 | 0.5790 | 0.4166 | 0.8191 | 0.5969 | 0.5130 |
| RandomForest | 0.8181 | 0.5994 | 0.4816 | 0.8263 | 0.6105 | 0.5549 |
| GradientBoosting | 0.8184 | 0.5973 | 0.4258 | **0.8312** | **0.6205** | 0.5038 |
| HistGradientBoosting | 0.8174 | **0.5995** | 0.4270 | 0.8297 | 0.6198 | 0.5053 |
| HistGB + sigmoid, threshold 0.34 | — | — | — | 0.8297 | 0.6199 | **0.5739** |

O HistGradientBoosting venceu o critério de seleção no treino por PR-AUC médio. O GradientBoosting teve a melhor discriminação pontual no holdout, mas essa informação não foi usada para trocar o candidato escolhido.

## Calibração

| Variante OOF | Brier | Log Loss |
|---|---:|---:|
| Sem calibração | 0.118284 | 0.382024 |
| Sigmoid | **0.118127** | **0.381495** |
| Isotonic | 0.118349 | 0.385664 |

Sigmoid trouxe uma melhoria pequena, consistente nas duas métricas, e foi selecionada. No holdout, a v3 obteve Brier `0.116140` e Log Loss `0.373771`, contra `0.119805` e `0.383456` da v2.

## Thresholds e negócio

Thresholds derivados exclusivamente de probabilidades OOF:

| Estratégia | Threshold | Precision OOF | Recall OOF | F1 OOF |
|---|---:|---:|---:|---:|
| Menor custo simulado | 0.17 | 0.4140 | 0.7411 | 0.5312 |
| High recall | 0.21 | 0.4514 | 0.6982 | 0.5483 |
| Equilibrado / maior F1 | **0.34** | 0.5992 | 0.5393 | **0.5676** |
| Default histórico | 0.50 | 0.6654 | 0.4331 | 0.5247 |
| High precision | 0.54 | 0.6757 | 0.4129 | 0.5126 |

O custo ilustrativo usa `FN=5` e `FP=1`; não representa valores financeiros reais. A v3 usa `0.34` em metadata, mas a aplicação continua apontando para v2 até promoção explícita.

## Holdout final da v3

Com threshold `0.34`: accuracy `0.8300`, precision `0.5872`, recall `0.5613`, F1 `0.5739`, ROC-AUC `0.8297` e PR-AUC `0.6199`. Matriz: TN `1431`, FP `161`, FN `179`, TP `229`.

A classe Stay obteve precision `0.8888`, recall `0.8989` e F1 `0.8938`. A classe Churn obteve precision `0.5872`, recall `0.5613` e F1 `0.5739`.

## Escolha e artefato

Decisão: **C — CREATE V3 AND RECOMMEND PROMOTION**. A melhoria de PR-AUC CV sobre a v2 foi de aproximadamente `0.0204`, superior ao desvio padrão da v2, e o holdout confirmou melhores PR-AUC, ROC-AUC, Brier, Log Loss, recall e F1. O custo é maior latência e um pequeno recuo de precision.

O artefato `models/v3/pipeline.joblib` é um `CalibratedClassifierCV` sobre o pipeline raw completo e aceita `predict(raw_df)` e `predict_proba(raw_df)`. A metadata registra parâmetros, calibração, CV, métricas e thresholds reais. A v2 continua sendo o default e champion operacional até promoção explícita.

## Resultados e limitações

Os CSVs em `reports/model-comparison/` contêm resultados de CV, holdout, buscas, classification reports, curvas ROC/PR, reliability diagram e análise de threshold. As três features atuais limitam a discriminação e tornam a Logistic Regression particularmente fraca. O custo de negócio é apenas uma simulação; a promoção deve incluir aceite de negócio e canary/monitoramento.
