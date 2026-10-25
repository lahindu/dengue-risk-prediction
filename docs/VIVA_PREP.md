# Questions I might get, and my answers

**Why this problem?** Dengue is the biggest mosquito disease in Sri Lanka. One week warning per
district gives time for fogging and clean-ups.

**Where is the data from?** Weekly Epidemiological Reports of the Epidemiology Unit, Ministry of
Health, collected in the denguedatahub project.

**Why a threshold for each district?** Colombo has about 173 cases a week on average and Mannar
about 3. One national number would make Colombo always high risk and Mannar never.

**What is data leakage and how did you avoid it?** Using information you would not have when
predicting. Next week's cases are only used to make the target, the threshold uses training
years only, and the split is by time.

**Why log transform?** The data is very skewed (10.2). Log makes it 0.2. log1p also works with 0.

**Why scale if trees don't need it?** Logistic Regression needs it, and it is inside Stacking.
Using one pipeline for everything is simpler.

**Missing values?** One Puttalam week filled by interpolation. Empty boxes in the app are filled
with the median by the pipeline.

**Why not a random split?** Weeks next to each other are almost the same (autocorrelation 0.79).
A random split would leak the future into training.

**Why PR-AUC and not accuracy?** Only 24% of weeks are high risk. PR-AUC focuses on finding the
high-risk weeks and does not depend on the 0.5 cut-off.

**What does recall 0.90 mean?** Of 843 real high-risk weeks in 2025-26, the model warned for 759.

**Bagging vs boosting?** Bagging (Random Forest): many trees on random samples, averaged, less
variance. Boosting (XGBoost, LightGBM, CatBoost): trees one after another, each fixes the errors
of the last, less bias.

**LightGBM vs XGBoost vs CatBoost?** LightGBM grows trees leaf by leaf (fast but can overfit, so
num_leaves=7). XGBoost grows level by level. CatBoost uses symmetric trees and ordered boosting,
more stable on noisy data.

**How does Stacking work?** The 4 models make predictions on folds they were not trained on
(cv=5). Then Logistic Regression learns how much to trust each model. Soft Voting just averages.

**Random Forest had 0.001 higher test PR-AUC, why Stacking?** I chose on 2024 before looking at
the test years. 0.001 is noise. Stacking also misses 25 fewer high-risk weeks.

**Most important parameter?** Class weights. Recall went from about 0.70 to 0.82-0.89.

**What is a SHAP value?** How much one feature pushed one prediction up or down compared to the
average prediction. Base value + all SHAP values = the prediction (0.389 + ... = 0.96).

**What is LIME?** It changes the input a little many times, sees what the model predicts, and fits
a small linear model around that one point. It is random, so I tested it with 10 seeds.

**SHAP vs LIME?** SHAP has a math guarantee (the values add up) and was more stable. LIME is
easier to read but changes between runs. They agree on the top 2 features.

**Why do SHAP and permutation importance disagree on month?** SHAP shows how much a feature moves
the prediction. Permutation shows how much accuracy we lose when it is shuffled. Month moves the
prediction but the recent cases already contain the season, so shuffling it costs little.

**Does "cases this week" cause next week's risk?** No. SHAP explains the model, not causes. Rain
and mosquitoes are the real causes and are not in the data.
