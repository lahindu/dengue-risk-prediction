# Prompt I used for the app

The assignment allows using ChatGPT or Gemini for the front end. This is the prompt I used.
After that I changed the code myself (tabs, SHAP chart, error message for empty input).

> I have a scikit-learn pipeline saved with pickle in models/model.pkl. It predicts if a
> Sri Lankan district will be high risk for dengue next week (1 or 0). models/model_info.json
> has the list of districts and the threshold of each district. In src/data_prep.py there is a
> function make_one_row(district, week_start, cases_now, cases_1, cases_2, cases_3,
> district_list, thresholds) that makes one input row.
>
> Make a simple Streamlit app with:
> - a dropdown for the district and a date box for the week start
> - 4 number boxes for cases this week, 1 week ago, 2 weeks ago and 3 weeks ago
>   (the older ones can be empty)
> - a Predict button that shows "HIGH RISK" in red or "Normal" in green with the probability
> - more tabs to show the charts in reports/figures and the tables reports/metrics.csv and
>   reports/validation_metrics.csv
> Keep the code simple with comments.

Then I asked:

> Add a SHAP bar chart under the prediction that shows why the model gave that answer.
> Use the functions make_shap_explainer and get_shap_values from src/explain.py.
