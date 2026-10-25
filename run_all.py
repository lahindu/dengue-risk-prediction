# Runs everything in order. Takes about 8 minutes.
#   python run_all.py
# (tuning is separate because it is slow: python -m src.tuning)

import warnings
warnings.filterwarnings("ignore")

from src import eda, train, charts, explain, predict

print("===== 1. EDA =====")
eda.main()
print("\n===== 2. Train and compare all models =====")
train.main()
print("\n===== 3. Comparison charts =====")
charts.main()
print("\n===== 4. SHAP and LIME =====")
explain.main()
print("\n===== 5. Inference on the left out week =====")
predict.main()
print("\nDone! Start the app with:  streamlit run app/app.py")
