import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import tensorflow as tf
from preprocessing import preprocess_single_input, preprocess_batch_data, validate_input_columns, create_sample_input

# Custom metric (ANN)
@tf.keras.utils.register_keras_serializable()
def f1_macro(y_true, y_pred):
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
    
    if len(y_true_one_hot.shape) > 2:
        y_true_one_hot = tf.squeeze(y_true_one_hot, axis=[1])

    def recall_m(y_true, y_pred):
        TP = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        Positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        recall = TP / (Positives + tf.keras.backend.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + tf.keras.backend.epsilon())
        return precision
    
    precision = precision_m(y_true_one_hot, y_pred)
    recall = recall_m(y_true_one_hot, y_pred)
    
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

CREDIT_LABELS = ['Poor', 'Standard', 'Good']
CREDIT_COLORS = ['#ff4757', '#ffa502', '#2ed573']

OCCUPATION_OPTIONS = ['Journalist', 'Accountant', 'Scientist', 'Manager', 'Media_Manager',
                     'Musician', 'Entrepreneur', 'Writer', 'Architect', 'Mechanic', 
                     'Doctor', 'Teacher', 'Lawyer', 'Developer', 'Engineer']

CREDIT_MIX_OPTIONS = ['Good', 'Standard', 'Bad']
PAYMENT_MIN_OPTIONS = ['No', 'Yes']
PAYMENT_BEHAVIOUR_OPTIONS = ['High_spent_Medium_value_payments', 'High_spent_Large_value_payments',
                            'Low_spent_Small_value_payments', 'Low_spent_Large_value_payments',
                            'High_spent_Small_value_payments', 'Low_spent_Medium_value_payments']

st.set_page_config(
    page_title="Credit Score Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    rf_model = None
    nn_model = None
    
    try:
        with st.spinner("Loading Random Forest model..."):
            try:
                rf_model = joblib.load('model/rf_optuna_model_fixed.joblib')
            except Exception as e1:
                st.warning(f"joblib loading failed: {str(e1)}")
                try:
                    with open('model/rf_optuna_model_fixed.joblib', 'rb') as f:
                        rf_model = pickle.load(f)
                except Exception as e2:
                    st.error(f"Both joblib and pickle loading failed for RF model: {str(e2)}")
                    
    except Exception as e:
        st.error(f"Error loading Random Forest model: {str(e)}")
        
    try:
        with st.spinner("Loading Neural Network model..."):
            nn_model = tf.keras.models.load_model(
                'model/best_ann_model.keras',
                custom_objects={'f1_macro': f1_macro}
            )
            
    except Exception as e:
        st.error(f"Error loading Neural Network model: {str(e)}")
        
    return rf_model, nn_model

def predict_credit_score(input_data, model_type="Random Forest"):
    try:
        rf_model, nn_model = load_models()
        
        if model_type == "Random Forest" and rf_model is not None:
            model = rf_model
        elif model_type == "Neural Network" and nn_model is not None:
            model = nn_model
        else:
            return None
        
        # Get predictions
        if model_type == "Random Forest":
            probabilities = model.predict_proba(input_data)[0]
            predicted_class = model.predict(input_data)[0]
        else:  # Neural Network
            probabilities = model.predict(input_data)[0]
            predicted_class = np.argmax(probabilities)
        
        return {
            'predicted_class': int(predicted_class),
            'predicted_label': CREDIT_LABELS[predicted_class],
            'probabilities': {label: float(prob) for label, prob in zip(CREDIT_LABELS, probabilities)}
        }
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def create_probability_chart(probabilities):
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = CREDIT_COLORS
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Credit Score Probabilities",
        xaxis_title="Credit Score Category",
        yaxis_title="Probability",
        showlegend=False,
        height=300
    )
    
    return fig

def create_batch_distribution_chart(results_df):
    score_counts = results_df['predicted_credit_score'].value_counts()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=score_counts.index,
            values=score_counts.values,
            marker_colors=[CREDIT_COLORS[CREDIT_LABELS.index(label)] for label in score_counts.index]
        )
    ])
    
    fig.update_layout(
        title="Credit Score Distribution",
        height=400
    )
    
    return fig

def main():
    st.title("Credit Score Prediction System")
    st.markdown("""
    This application uses machine learning models to predict credit scores based on financial and personal information.
    The system can classify credit scores into 3 categories: **Poor**, **Standard**, or **Good**.
    """)
    
    # Load models
    with st.container():
        rf_model, nn_model = load_models()
    
    if rf_model is None and nn_model is None:
        st.error("Failed to load any models. Please check if model files exist in the 'model/' directory.")
        st.stop()
    elif rf_model is None:
        st.warning("Random Forest model failed to load. Only Neural Network will be available.")
        st.session_state.available_models = ["Neural Network"]
    elif nn_model is None:
        st.warning("Neural Network model failed to load. Only Random Forest will be available.")
        st.session_state.available_models = ["Random Forest"]
    else:
        st.session_state.available_models = ["Random Forest", "Neural Network"]
    
    # === Sidebar ===
    with st.sidebar:
        st.header("🤖 Model Selection")
        
        model_type = st.selectbox(
            "Choose Model:",
            st.session_state.get('available_models', ["Random Forest", "Neural Network"]),
            help="Select which model to use for predictions"
        )
        
        st.markdown("---")
        st.header("📊 Model Performance")
        
        st.success("Metrics: F1-Macro Score")
        if model_type == "Random Forest":
            st.info("""
            **🌲 Random Forest Model**
            - **Train:** 85.81%
            - **Test:** 76.97%
            - **Algorithm:** Optimized Random Forest
            - **Optimization:** Optuna hyperparameter tuning
            """)
        else:
            st.info("""
            **🧠 Neural Network Model**
            - **Train:** 85.94%
            - **Test:** 78.36%
            - **Architecture:** Deep Neural Network
            - **Framework:** TensorFlow/Keras
            """)
        
        st.markdown("---")
        st.header("⚙️ Prediction Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Minimum confidence score to highlight predictions"
        )
        
        st.markdown("---")
        st.header("💡 Sample Data")
        
        st.markdown("**Poor Credit Score:**")
        st.code("Low income, high debt, many delayed payments")
        
        st.markdown("**Standard Credit Score:**")
        st.code("Moderate income, some debt, occasional delays")
        
        st.markdown("**Good Credit Score:**")
        st.code("High income, low debt, no payment delays")
    
    # === Main content tabs ===
    tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction"])
    
    with tab1:
        st.header("🔍 Single Credit Score Prediction")
        st.markdown("Enter customer information to predict their credit score.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            age = st.slider("Age", min_value=14, max_value=70, value=33)
            occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS, index=OCCUPATION_OPTIONS.index('Lawyer'))
            annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=37182.62, step=1000.0)
            
            st.subheader("Banking Information")
            num_bank_accounts = st.slider("Number of Bank Accounts", min_value=0, max_value=12, value=5)
            num_credit_cards = st.slider("Number of Credit Cards", min_value=0, max_value=12, value=5)
            interest_rate = st.slider("Interest Rate (%)", min_value=0, max_value=35, value=13)
            
            st.subheader("Loan Information")
            num_loans = st.slider("Number of Loans", min_value=0, max_value=10, value=3)
            delay_from_due_date = st.slider("Days Delayed from Due Date", min_value=0, max_value=70, value=18)
            num_delayed_payments = st.slider("Number of Delayed Payments", min_value=0, max_value=25, value=14)
            
            st.subheader("Credit Information")
            changed_credit_limit = st.slider("Changed Credit Limit ($)", min_value=0.0, max_value=40.0, value=9.36, step=0.1)
            num_credit_inquiries = st.slider("Number of Credit Inquiries", min_value=0, max_value=20, value=5)
            outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, value=1161.10, step=100.0)
            
        with col2:
            st.subheader("Financial Ratios")
            credit_utilization_ratio = st.slider("Credit Utilization Ratio (%)", min_value=20.0, max_value=50.0, value=32.30, step=0.1)
            total_emi_per_month = st.number_input("Total EMI per Month ($)", min_value=0.0, value=66.49, step=10.0, max_value=1000.0)
            amount_invested_monthly = st.number_input("Amount Invested Monthly ($)", min_value=0.0, value=127.76, step=10.0, max_value=15000.0)
            monthly_balance = st.number_input("Monthly Balance ($)", value=344.82, step=50.0, max_value=2000.0)
            
            st.subheader("Credit History")
            credit_history_years = st.slider("Credit History (Years)", min_value=0, max_value=35, value=18)
            credit_history_months = st.slider("Additional Months", min_value=0, max_value=11, value=2)
            credit_history_age = f"{credit_history_years} Years and {credit_history_months} Months"
            
            st.subheader("Behavioral Information")
            credit_mix = st.selectbox("Credit Mix", CREDIT_MIX_OPTIONS, index=CREDIT_MIX_OPTIONS.index('Standard'))
            payment_of_min_amount = st.selectbox("Payment of Minimum Amount", PAYMENT_MIN_OPTIONS, index=PAYMENT_MIN_OPTIONS.index('Yes'))
            payment_behaviour = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOUR_OPTIONS, index=PAYMENT_BEHAVIOUR_OPTIONS.index('Low_spent_Small_value_payments'))
        
        # Prediction button
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            predict_button = st.button(
                "🔮 Predict Credit Score",
                type="primary",
                use_container_width=True
            )
        
        if predict_button:
            input_dict = {
                'Age': age,
                'Occupation': occupation,
                'Annual_Income': annual_income,
                'Num_Bank_Accounts': num_bank_accounts,
                'Num_Credit_Card': num_credit_cards,
                'Interest_Rate': interest_rate,
                'Num_of_Loan': num_loans,
                'Delay_from_due_date': delay_from_due_date,
                'Num_of_Delayed_Payment': num_delayed_payments,
                'Changed_Credit_Limit': changed_credit_limit,
                'Num_Credit_Inquiries': num_credit_inquiries,
                'Outstanding_Debt': outstanding_debt,
                'Credit_Utilization_Ratio': credit_utilization_ratio,
                'Credit_History_Age': credit_history_age,
                'Total_EMI_per_month': total_emi_per_month,
                'Amount_invested_monthly': amount_invested_monthly,
                'Monthly_Balance': monthly_balance,
                'Credit_Mix': credit_mix,
                'Payment_of_Min_Amount': payment_of_min_amount,
                'Payment_Behaviour': payment_behaviour
            }
            
            with st.spinner("Processing prediction..."):
                try:
                    preprocessed_data = preprocess_single_input(input_dict)
                    
                    result = predict_credit_score(preprocessed_data, model_type)
                    
                    if result:
                        st.markdown("---")
                        
                        # Display results
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("📋 Prediction Results")
                            
                            predicted_label = result['predicted_label']
                            max_probability = max(result['probabilities'].values())
                            
                            # Color based on credit score
                            color_map = {"Good": "green", "Standard": "orange", "Poor": "red"}
                            color = color_map[predicted_label]
                            
                            st.markdown(f"**Predicted Credit Score:** :{color}[{predicted_label}]")
                            st.markdown(f"**Confidence:** {max_probability:.3f}")
                            st.markdown(f"**Model Used:** {model_type}")
                            
                            if max_probability >= confidence_threshold:
                                st.success(f"High confidence prediction! (≥ {confidence_threshold})")
                            else:
                                st.warning(f"Low confidence prediction (< {confidence_threshold})")
                        
                        with col2:
                            st.subheader("📈 Probability Scores")
                            fig = create_probability_chart(result['probabilities'])
                            st.plotly_chart(fig, use_container_width=True)
                            
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
    
    with tab2:
        st.header("📊 Batch Credit Score Prediction")
        st.markdown("Upload a CSV file containing customer data for batch prediction.")
        
        # Download sample CSV
        with open("resource/sample.csv", "rb") as f:
            sample_csv_data = f.read()
        
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=sample_csv_data,
            file_name="credit_score_sample.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type="csv",
            help="Upload CSV file with customer data for batch prediction"
        )
        
        if uploaded_file is not None:
            try:
                try:
                    string_data = uploaded_file.getvalue().decode('utf-8')
                    df = pd.read_csv(StringIO(string_data))
                except UnicodeDecodeError:
                    string_data = uploaded_file.getvalue().decode('cp1252')
                    df = pd.read_csv(StringIO(string_data))
                
                st.success(f"File uploaded successfully! Found {len(df)} rows.")
                
                # Validate columns
                is_valid, missing_cols = validate_input_columns(df)
                
                if not is_valid:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.info("Please ensure your CSV contains all required columns. Download the sample template for reference.")
                else:
                    # Show data preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Batch prediction button
                    predict_batch_button = st.button(
                        "🚀 Predict All Credit Scores",
                        type="primary",
                        use_container_width=True
                    )
                    
                    if predict_batch_button:
                        st.info(f"Processing {len(df)} records with {model_type} model...")
                        
                        try:
                            with st.spinner("Processing batch predictions..."):
                                # Preprocess batch data
                                X_processed, df_processed = preprocess_batch_data(df)
                                
                                # Make predictions
                                results = []
                                progress_bar = st.progress(0)
                                
                                for i in range(len(X_processed)):
                                    input_data = X_processed[i:i+1]
                                    result = predict_credit_score(input_data, model_type)
                                    
                                    if result:
                                        results.append({
                                            'row_index': i,
                                            'predicted_credit_score': result['predicted_label'],
                                            'confidence': max(result['probabilities'].values()),
                                            'prob_poor': result['probabilities']['Poor'],
                                            'prob_standard': result['probabilities']['Standard'],
                                            'prob_good': result['probabilities']['Good']
                                        })
                                    
                                    progress_bar.progress((i + 1) / len(X_processed))
                                
                                if results:
                                    results_df = pd.DataFrame(results)
                                    
                                    # Combine with original data
                                    final_results = df.copy()
                                    final_results['predicted_credit_score'] = results_df['predicted_credit_score']
                                    final_results['confidence'] = results_df['confidence']
                                    final_results['prob_poor'] = results_df['prob_poor']
                                    final_results['prob_standard'] = results_df['prob_standard']
                                    final_results['prob_good'] = results_df['prob_good']
                                    
                                    # Display results
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.subheader("Credit Score Distribution")
                                        fig = create_batch_distribution_chart(results_df)
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    with col2:
                                        st.subheader("Summary Statistics")
                                        score_counts = results_df['predicted_credit_score'].value_counts()
                                        for score, count in score_counts.items():
                                            percentage = (count / len(results_df)) * 100
                                            st.metric(score, f"{count} ({percentage:.1f}%)")
                                        
                                        avg_confidence = results_df['confidence'].mean()
                                        st.metric("Average Confidence", f"{avg_confidence:.3f}")
                                    
                                    # Detailed results table
                                    st.subheader("📊 Detailed Results")
                                    st.dataframe(final_results, use_container_width=True)
                                    
                                    # Download results
                                    csv = final_results.to_csv(index=False)
                                    st.download_button(
                                        label="💾 Download Results as CSV",
                                        data=csv,
                                        file_name="credit_score_predictions.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                    
                        except Exception as e:
                            st.error(f"Error during batch prediction: {str(e)}")
                            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # === Footer ===
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Credit Score Prediction System | Models: Random Forest & Neural Network | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()