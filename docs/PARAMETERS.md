# Parameters and preprocessing

I did not use default parameters. I picked starting values from what I know about the data and
then tested the important ones with `GridSearchCV` and `TimeSeriesSplit` (3 folds) on the
training years only (`src/tuning.py`, results in `reports/tuning_results.csv`).

I used F1 to compare the settings. If two settings were almost the same (less than 0.005
difference), I picked the one with better PR-AUC or the simpler/faster one.

Things about the data that helped me choose:
- around 24,000 training rows and 11 features, so the trees don't need to be very deep
- only about 24% of weeks are high risk (3 normal : 1 high risk), so I need class weights
- weekly counts are noisy, so leaves should not be too small
- missing a high-risk week is worse than a false alarm, so recall matters

## Baselines

| Model | Setting | Why |
|---|---|---|
| Persistence rule | high risk if this week > district 75th percentile | what you can do without ML |
| Logistic Regression | `class_weight="balanced"`, one-hot districts | rare class; district is a label not a number |
| Decision Tree | `max_depth=6`, `min_samples_leaf=50` | one deep tree overfits |

## Random Forest (16 combinations tested)

| Parameter | Default | Mine | Why |
|---|---|---|---|
| n_estimators | 100 | 300 | more trees = more stable |
| max_depth | None | 20 | 20 was a bit better than 10 |
| min_samples_leaf | 1 | 10 | 10 was better than 3 every time |
| max_features | sqrt | sqrt | tested 0.5 too, same score, sqrt is faster |
| class_weight | None | balanced | the most important one, recall went up a lot |

## XGBoost (12 combinations)

| Parameter | Default | Mine | Why |
|---|---|---|---|
| n_estimators | 100 | 400 | many small steps |
| max_depth | 6 | 3 | tested 3, 5, 7: deeper was always worse |
| learning_rate | 0.3 | 0.05 | 0.05 better than 0.1 |
| subsample, colsample_bytree | 1 | 0.8 | some randomness to reduce overfitting |
| min_child_weight | 1 | 3 | a split needs more data |
| scale_pos_weight | 1 | 3 | about 3 normal weeks per high-risk week |

## LightGBM (12 combinations)

| Parameter | Default | Mine | Why |
|---|---|---|---|
| num_leaves | 31 | 7 | tested 7, 15, 31: F1 0.693, 0.687, 0.679 |
| learning_rate | 0.1 | 0.05 | better every time |
| n_estimators | 100 | 400 | more rounds for the smaller learning rate |
| min_child_samples | 20 | 40 | noisy data |
| subsample (+ subsample_freq=1), colsample_bytree | 1 | 0.8 | like XGBoost |
| reg_lambda | 0 | 1 | a little regularisation |
| scale_pos_weight | 1 | 3 | class balance |

## CatBoost (8 combinations)

| Parameter | Default | Mine | Why |
|---|---|---|---|
| depth | 6 | 4 | 4 and 6 almost the same, 4 had better PR-AUC and is faster |
| learning_rate | auto | 0.05 | better than 0.1 |
| iterations | 1000 | 500 | no improvement after about 500 |
| l2_leaf_reg | 3 | 5 | more regularisation |
| auto_class_weights | None | SqrtBalanced | best F1 (0.696), softer than "Balanced" |

## Voting and Stacking

| Model | Setting | Why |
|---|---|---|
| Soft Voting | RF + XGBoost + LightGBM + CatBoost, voting="soft" | average of the probabilities |
| Stacking | same 4 models, final model Logistic Regression (C=0.5, balanced), cv=5 | a simple final model so it does not overfit |

## Preprocessing

| Step | What | Why |
|---|---|---|
| Missing values | checked nulls, duplicates and missing weeks | no nulls, but Puttalam missed 1 week |
| Missing week | linear interpolation inside the district | cases change smoothly week to week |
| Empty inputs in the app | median imputer in the pipeline | median because the data is skewed |
| Log transform | log(1 + x) on the 5 case columns | skewness 10.2 -> 0.2 |
| Threshold features | log(1+cases) - log(1+threshold) | tells the model how far above normal; RF PR-AUC 0.776 -> 0.848 |
| Scaling | StandardScaler | Logistic Regression needs it (also inside Stacking), trees don't care |
| District | number for trees, one-hot for Logistic Regression | it is a name, not a size |
| Removed rows | first 3 weeks of each district | no lag values |
| No data leakage | threshold from training years only, next week's cases never used as input, time split | so the test is fair |
