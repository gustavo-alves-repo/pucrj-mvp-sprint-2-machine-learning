# MVP — Sprint 02 - Machine Learning - Previsão de vitória das **BRANCAS** em partidas do Chess.com (win vs not_win)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/gustavo-alves-repo/pucrj-mvp-sprint-2-machine-learning/blob/main/mvp_sprint_2_machine_learning.ipynb)

**Autor:** Gustavo Alves  
**Matrícula:** 4052025001911  
**Notebook principal:** [mvp_sprint_2_machine_learning.ipynb](https://colab.research.google.com/github/gustavo-alves-repo/pucrj-mvp-sprint-2-machine-learning/blob/main/mvp_sprint_2_machine_learning.ipynb)  
**Artefato:** `modelo_win_notwin.joblib`

## Objetivo
Prever, **antes do início da partida**, se as **BRANCAS** vão **vencer (win)** ou **não vencer (not_win = draw ou lose)** usando somente informações **pré-jogo** (ratings, diferença de rating, classe de tempo, etc.).

## Dados
- **Fonte:** [Chess Game Dataset (Kaggle)](https://www.kaggle.com/datasets/adityajha1504/chesscom-user-games-60000-games)  
- Trabalho com ~60k jogos públicos do Chess.com (EN-US), textos do notebook em PT-BR.  
- Para execução reprodutível, o CSV é lido diretamente de um link do GitHub.

## Escopo & formulação
- **Tarefa:** Classificação **binária** (win vs not_win).
- **Positiva (1):** win (vitória das BRANCAS).  
- **Negativa (0):** not_win (empate ou derrota das BRANCAS).
- **Features:** `rating_diff`, `rated` (0/1), dummies de `time_class` (`tc_bullet`, `tc_blitz`, `tc_rapid`, `tc_daily`).
- **Filtros:** apenas `rules == 'chess'` (xadrez clássico).
- **Seed global:** `SEED = 42`.

## Abordagem de modelagem
1. **EDA resumida** para entender distribuição de resultados e o impacto de `rating_diff`.
2. **Pré-processamento leve** (one-hot em `time_class`).
3. **Holdout 70/30 estratificado** + **Cross-Validation** nos 70% para comparar modelos.
4. **Baselines:**
   - **Majoritária** (classe constante) — referência mínima.
   - **Elo** (probabilidade de vitória via fórmula logística do Elo a partir de `rating_diff`).
5. **Modelos candidatos:** LR, RF, KNN, CART, NB e SVM\*
6. **Critérios de seleção:** menor **LogLoss** na CV; decisão operacional com **F1**.
7. **Threshold (τ):** eu **fixei τ = 0,50** (equilíbrio precisão/recall). (O notebook mostra como varrer τ, se necessário.)

> \***Observação:** Não consegui rodar o SVM, demora muito e não finaliza. O código possui uma flag binária para saber se deve rodar ou não o SVM. O MVP foi feito sem o SVM. O código roda em menos de 2 minutos sem o SVM.

## Resultados (teste, τ = 0,50)

**Modelo escolhido:** **Logistic Regression (LR)**

| Modelo | Precision | Recall | F1 | ROC-AUC | PR-AUC | LogLoss |
|---|---:|---:|---:|---:|---:|---:|
| **LR (τ=0,50)** | ~0.695 | ~0.695 | ~0.695 | ~0.763 | ~0.738 | **~0.617** |
| Elo | ~0.691 | ~0.703 | ~0.697 | ~0.763 | ~0.738 | ~0.624 |
| Majoritária | ~0.000 | ~0.000 | ~0.000 | 0.500 | 0.498 | ~17.96 |

**Leitura:** o **Elo já é um baseline muito forte** neste problema. A **LR empata em discriminação** (ROC/PR) **e F1**, mas **melhora um pouco o LogLoss**, ou seja, entrega **probabilidades mais bem calibradas**.

> **Observação:** no notebook eu também mostro as curvas **ROC** e **Precision-Recall** e a **matriz de confusão** com τ = 0,50.

## Como reproduzir

### Opção A — Google Colab)
1. Abra o notebook `mvp_sprint_2_machine_learning.ipynb` no Colab.  
2. Execute **Runtime → Run all**.  
3. O dataset é lido via URL do meu GitHub público; não precis configurar nada.

### Opção B — Local (ex.: VS Code)
```bash
python -m venv .venv
source .venv/Scripts/activate  # Mac: .venv/bin/activate
pip install -r requirements.txt
jupyter notebook  # ou jupyter lab
