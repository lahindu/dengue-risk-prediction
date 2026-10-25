# Video plan (max 5 minutes)

Before recording: run `streamlit run app/app.py`, open the notebook or VS Code, zoom in
(Ctrl +) so the text is easy to read.

| Time | Show | Say |
|---|---|---|
| 0:00-0:30 | README | Problem: predict if a district will be high risk for dengue next week, so officers can act early. Data: Ministry of Health weekly reports, 26 districts, 2006-2026. |
| 0:30-1:15 | App -> Data tab | EDA: very skewed so I use log; two monsoon peaks so week and month; strong week to week link so lag features and a persistence rule to compare with; 2025-26 had much more dengue. |
| 1:15-2:30 | App -> Model comparison tab | Time split: train 2007-2023, choose on 2024, test on 2025-26. 9 models from simple rule to Stacking, tuned with TimeSeriesSplit. Stacking was best on 2024, recall 0.90 on the test years, 131 more high-risk weeks found than the simple rule. |
| 2:30-3:45 | App -> Predict tab: Colombo, 328 / 300 / 279 / 240 | 96% high risk, real answer 308 cases so correct. SHAP: the 4 week average and this week being above Colombo's normal level add about 24 and 23 points. |
| 3:45-4:40 | App -> SHAP and LIME tab | LIME agrees on the top 2. LIME changes a bit with different seeds. Wrong predictions are close to the threshold or at the end of a wave. |
| 4:40-5:00 | RESULTS.md limitations | No rain data yet; SHAP explains the model, not the cause of dengue. Thank you. |
