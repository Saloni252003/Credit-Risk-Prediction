import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("Credit Risk Dataset - Interactive EDA Dashboard")


@st.cache_data
def load_data():
    return pd.read_csv(r"C:/Users/ASUS/OneDrive/Documents/credit_risk_dataset_1.xls")

df = load_data()

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


@st.cache_resource
def load_model():
    with open(r"C:/Users/ASUS/OneDrive/Documents/RF_model (1).pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


OHE_FEATURE_NAMES = [
    "home_MORTGAGE", "home_OTHER", "home_OWN", "home_RENT",
    "intent_DEBTCONSOLIDATION", "intent_EDUCATION", "intent_HOMEIMPROVEMENT",
    "intent_MEDICAL", "intent_PERSONAL", "intent_VENTURE",
    "grade_A", "grade_B", "grade_C", "grade_D", "grade_E", "grade_F", "grade_G",
    "default_on_file_N", "default_on_file_Y",]


def build_input(cb_person_default_on_file, loan_intent, loan_grade, person_home_ownership):
    """
    Builds the exact 19-feature OHE array the saved model was trained on.
    Numerical columns (age, income, etc.) are NOT passed — the saved model
    was trained without them.
    """
    all_home    = ["MORTGAGE", "OTHER", "OWN", "RENT"]
    all_intents = ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
                   "MEDICAL", "PERSONAL", "VENTURE"]
    all_grades  = ["A", "B", "C", "D", "E", "F", "G"]
    all_default = ["N", "Y"]

    values = []

    for val in all_home:
        values.append(1 if person_home_ownership == val else 0)

    for val in all_intents:
        values.append(1 if loan_intent == val else 0)

    for val in all_grades:
        values.append(1 if loan_grade == val else 0)

    for val in all_default:
        values.append(1 if cb_person_default_on_file == val else 0)

    return pd.DataFrame([values], columns=OHE_FEATURE_NAMES)


with st.sidebar:
    st.header("📊 Navigation")

    page = st.radio(
        "Go to Section",
        ["📋 Dataset Overview", "👤 Borrower Analysis", "📈 Univariate Analysis",
         "📈 Bivariate Analysis", "🤖 Predictions"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if page in ["📈 Univariate Analysis", "📈 Bivariate Analysis"]:
        st.subheader("Analysis Controls")

        if page == "📈 Univariate Analysis":
            uni_type = st.radio("Variable Type", ["Numerical", "Categorical"], horizontal=True)
            if uni_type == "Numerical":
                selected_num_cols = st.multiselect(
                    "Select Numerical Columns", options=num_cols,
                    default=['person_age', 'person_income', 'loan_amnt', 'loan_int_rate'])
            else:
                selected_cat_cols = st.multiselect(
                    "Select Categorical Columns", options=cat_cols,
                    default=['person_home_ownership', 'loan_intent', 'loan_grade'])

        elif page == "📈 Bivariate Analysis":
            biv_x = st.selectbox("X Column", df.columns.tolist())
            remaining = [col for col in df.columns if col != biv_x]
            biv_y = st.selectbox("Y Column", remaining)

    if page == "🤖 Predictions":
        st.subheader("⚙️ Input Features")

       
        person_age                 = st.number_input("Age", 18, 80, 30)
        person_income              = st.number_input("Annual Income ($)", 0, 10_000_000, 50000, step=1000)
        person_emp_length          = st.number_input("Employment Length (yrs)", 0.0, 60.0, 3.0, step=0.5)
        loan_amnt                  = st.number_input("Loan Amount ($)", 500, 100000, 10000, step=500)
        loan_int_rate              = st.slider("Interest Rate (%)", 5.0, 25.0, 12.0, step=0.1)
        loan_percent_income        = st.slider("Loan % of Income", 0.0, 1.0, 0.2, step=0.01)
        cb_person_cred_hist_length = st.number_input("Credit History Length (yrs)", 0, 30, 5)

        cb_person_default_on_file  = st.selectbox("Previous Default on File?", ["N", "Y"])
        loan_intent                = st.selectbox("Loan Intent",
            ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        loan_grade                 = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        person_home_ownership      = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

        predict_btn = st.button("🔍 Predict Default Risk", use_container_width=True)



if page == "📋 Dataset Overview":
    st.header("📋 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Rows", f"{df.shape[0]:,}")
    with c2: st.metric("Default Rate", f"{df['loan_status'].mean()*100:.1f}%")
    with c3: st.metric("Numerical Columns", len(num_cols))
    with c4: st.metric("Categorical Columns", len(cat_cols))
    st.subheader("Data Preview")
    st.dataframe(df.head(15), use_container_width=True)

elif page == "👤 Borrower Analysis":
    st.header("👤 Borrower Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="person_age", color="loan_status",
                           title="Age Distribution by Loan Status")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x="loan_status", y="person_income", points=False,
                     title="Income by Loan Status")
        st.plotly_chart(fig, use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(df, x="loan_status", y="person_emp_length", points=False,
                     title="Employment Length by Loan Status")
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.histogram(df, x="cb_person_cred_hist_length", color="loan_status",
                           title="Credit History Length by Loan Status")
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Univariate Analysis":
    st.header("📈 Univariate Analysis")
    if 'uni_type' in locals():
        if uni_type == "Numerical" and 'selected_num_cols' in locals():
            for col in selected_num_cols:
                st.subheader(col)
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(df, x=col, nbins=30, title=f"Histogram of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.box(df, y=col, points=False, title=f"Box Plot of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                st.divider()
        elif uni_type == "Categorical" and 'selected_cat_cols' in locals():
            for col in selected_cat_cols:
                st.subheader(col)
                vc = df[col].value_counts().reset_index()
                vc.columns = [col, "Count"]
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(vc, x=col, y="Count", title=f"Bar Chart - {col}")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.pie(vc, names=col, values="Count", title=f"Pie Chart - {col}")
                    st.plotly_chart(fig, use_container_width=True)
                st.divider()

elif page == "📈 Bivariate Analysis":
    st.header("📈 Bivariate Analysis")
    st.subheader(f"{biv_x} vs {biv_y}")
    x_is_num = biv_x in num_cols
    y_is_num = biv_y in num_cols
    if x_is_num and y_is_num:
        fig = px.scatter(df, x=biv_x, y=biv_y, trendline="ols",
                         color="loan_status" if "loan_status" in df.columns else None)
    else:
        fig = px.box(df, x=biv_x, y=biv_y, points=False) if y_is_num else px.box(df, x=biv_y, y=biv_x, points=False)
    st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Predictions":
    st.header("🤖 Loan Default Risk Prediction")

    if not model_loaded:
        st.error(f"⚠️ Could not load model.\n\nError: {model_error}")
    else:
        st.info("Fill in the borrower details in the **sidebar** and click **Predict Default Risk**.")

        input_summary = {
            "Feature": ["Age", "Annual Income", "Employment Length", "Loan Amount",
                        "Interest Rate", "Loan % of Income", "Credit History Length",
                        "Previous Default", "Loan Intent", "Loan Grade", "Home Ownership"],
            "Value": [
                person_age, f"${person_income:,}", f"{person_emp_length} yrs",
                f"${loan_amnt:,}", f"{loan_int_rate}%", f"{loan_percent_income:.0%}",
                f"{cb_person_cred_hist_length} yrs", cb_person_default_on_file,
                loan_intent, loan_grade, person_home_ownership
            ]
        }
        st.subheader("📝 Current Input Summary")
        st.dataframe(pd.DataFrame(input_summary), use_container_width=True, hide_index=True)

        if predict_btn:
            # Only pass the 4 categorical inputs — model was trained on 19 OHE cols only
            input_data = build_input(
                cb_person_default_on_file, loan_intent,
                loan_grade, person_home_ownership
            )

            with st.expander("🔬 Debug: columns sent vs model expectations"):
                st.write(f"**Columns we're sending:** {input_data.shape[1]}")
                st.write(f"**Model expects:** {model.n_features_in_} features")
                if input_data.shape[1] == model.n_features_in_:
                    st.success("✅ Feature count matches!")
                else:
                    st.error(f"❌ Mismatch: sending {input_data.shape[1]}, model expects {model.n_features_in_}")
                st.dataframe(input_data.T.rename(columns={0: "Value"}))

            try:
                prediction   = model.predict(input_data)[0]
                proba        = model.predict_proba(input_data)[0]
                default_prob = proba[1] * 100
                safe_prob    = proba[0] * 100

                st.markdown("---")
                st.subheader("🎯 Prediction Result")
                col1, col2, col3 = st.columns(3)
                with col1:
                    label = "🔴 High Risk — Default" if prediction == 1 else "🟢 Low Risk — No Default"
                    st.metric("Prediction", label)
                with col2:
                    st.metric("Default Probability", f"{default_prob:.1f}%")
                with col3:
                    st.metric("Safe Probability", f"{safe_prob:.1f}%")

                st.markdown("#### Risk Probability Breakdown")
                gauge_df = pd.DataFrame({
                    "Outcome": ["No Default", "Default"],
                    "Probability": [safe_prob, default_prob]
                })
                fig = px.bar(gauge_df, x="Probability", y="Outcome", orientation="h",
                             color="Outcome",
                             color_discrete_map={"No Default": "#2ecc71", "Default": "#e74c3c"},
                             text_auto=".1f", range_x=[0, 100])
                fig.update_layout(showlegend=False, xaxis_title="Probability (%)", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

                if hasattr(model, "feature_importances_"):
                    st.markdown("#### 📊 Feature Importances")
                    feat_imp = pd.DataFrame({
                        "Feature":    OHE_FEATURE_NAMES,
                        "Importance": model.feature_importances_
                    }).sort_values("Importance", ascending=True)
                    fig2 = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                                  title="Feature Importance (top drivers of prediction)",
                                  color="Importance", color_continuous_scale="Blues")
                    fig2.update_layout(height=500)
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown("---")
                if prediction == 1:
                    st.error("⚠️ **High Default Risk Detected.**\n\nThis borrower profile suggests "
                             "elevated credit risk. Consider reviewing loan grade, interest rate, "
                             "and income-to-loan ratio before approval.")
                else:
                    st.success("✅ **Low Default Risk.**\n\nThis borrower profile appears creditworthy. "
                               "Standard approval procedures may apply.")

            except Exception as e:
                st.error(f"❌ Prediction failed.\n\n**Error:** {e}")
                st.warning("Open the **Debug** expander above to see which columns are missing or extra.")