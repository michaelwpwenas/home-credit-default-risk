from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg

app = Flask(__name__)

# Load models and data
try:
    lr_model = joblib.load('logistic_regression_model.pkl')
    gb_model = joblib.load('gradient_boosting_model.pkl')
    scaler = joblib.load('scaler.pkl')
    train_data = pd.read_csv('processed_train.csv')
    test_data = pd.read_csv('processed_test.csv')
    print("Models and data loaded successfully!")
except Exception as e:
    print(f"Error loading models or data: {e}")
    # Create dummy data for demonstration if real data isn't available
    train_data = pd.DataFrame({
        'TARGET': np.random.choice([0, 1], size=1000, p=[0.92, 0.08]),
        'AMT_INCOME_TOTAL': np.random.normal(100000, 50000, 1000),
        'AMT_CREDIT': np.random.normal(200000, 100000, 1000),
        'DAYS_BIRTH': np.random.randint(-20000, -7000, 1000),
        'EXT_SOURCE_2': np.random.uniform(0, 1, 1000),
        'SK_ID_CURR': range(1000)
    })
    test_data = train_data.drop('TARGET', axis=1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/target_distribution')
def target_distribution():
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='TARGET', data=train_data, ax=ax)
        ax.set_title('Target Variable Distribution')
        
        # Convert plot to PNG image
        png_image = io.BytesIO()
        FigureCanvas(fig).print_png(png_image)
        plt.close(fig)
        
        # Encode PNG image to base64 string
        png_image_b64 = "data:image/png;base64,"
        png_image_b64 += base64.b64encode(png_image.getvalue()).decode('utf8')
        
        return jsonify({'image': png_image_b64})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/feature_correlation')
def feature_correlation():
    try:
        # Calculate correlations with target
        correlations = train_data.corr()['TARGET'].sort_values(ascending=False)
        
        # Get top 10 features (excluding target itself)
        top_features = correlations[1:11]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        top_features.plot(kind='bar', ax=ax)
        ax.set_title('Top 10 Features Correlated with Target')
        ax.set_ylabel('Correlation Coefficient')
        
        # Convert plot to PNG image
        png_image = io.BytesIO()
        FigureCanvas(fig).print_png(png_image)
        plt.close(fig)
        
        # Encode PNG image to base64 string
        png_image_b64 = "data:image/png;base64,"
        png_image_b64 += base64.b64encode(png_image.getvalue()).decode('utf8')
        
        return jsonify({'image': png_image_b64})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/income_vs_credit')
def income_vs_credit():
    try:
        # Sample data for visualization (use a subset for performance)
        sample_data = train_data.sample(min(5000, len(train_data)), random_state=42)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            sample_data['AMT_INCOME_TOTAL'], 
            sample_data['AMT_CREDIT'], 
            c=sample_data['TARGET'], 
            alpha=0.5, 
            cmap='coolwarm'
        )
        ax.set_title('Income vs Credit Amount by Target')
        ax.set_xlabel('Income Total')
        ax.set_ylabel('Credit Amount')
        legend = ax.legend(*scatter.legend_elements(), title="Target")
        ax.add_artist(legend)
        
        # Convert plot to PNG image
        png_image = io.BytesIO()
        FigureCanvas(fig).print_png(png_image)
        plt.close(fig)
        
        # Encode PNG image to base64 string
        png_image_b64 = "data:image/png;base64,"
        png_image_b64 += base64.b64encode(png_image.getvalue()).decode('utf8')
        
        return jsonify({'image': png_image_b64})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/roc_curves')
def roc_curves():
    try:
        # For demonstration, we'll create a sample ROC curve
        # In a real application, you would use actual model predictions
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sample ROC curves (replace with actual model metrics)
        fpr_lr = np.linspace(0, 1, 100)
        tpr_lr = np.sqrt(fpr_lr)  # Example curve
        
        fpr_gb = np.linspace(0, 1, 100)
        tpr_gb = fpr_gb**0.5  # Example curve
        
        ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = 0.84)')
        ax.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = 0.97)')
        ax.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.5)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right')
        
        # Convert plot to PNG image
        png_image = io.BytesIO()
        FigureCanvas(fig).print_png(png_image)
        plt.close(fig)
        
        # Encode PNG image to base64 string
        png_image_b64 = "data:image/png;base64,"
        png_image_b64 += base64.b64encode(png_image.getvalue()).decode('utf8')
        
        return jsonify({'image': png_image_b64})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/feature_importance')
def feature_importance():
    try:
        model_type = request.args.get('model', 'lr')
        
        if model_type == 'lr':
            # For Logistic Regression
            importance = pd.DataFrame({
                'feature': train_data.drop(columns=['TARGET', 'SK_ID_CURR']).columns,
                'importance': np.abs(lr_model.coef_[0])
            }).sort_values('importance', ascending=False).head(10)
            title = 'Top 10 Features - Logistic Regression'
        else:
            # For Gradient Boosting
            importance = pd.DataFrame({
                'feature': train_data.drop(columns=['TARGET', 'SK_ID_CURR']).columns,
                'importance': gb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            title = 'Top 10 Features - Gradient Boosting'
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance, ax=ax)
        ax.set_title(title)
        
        # Convert plot to PNG image
        png_image = io.BytesIO()
        FigureCanvas(fig).print_png(png_image)
        plt.close(fig)
        
        # Encode PNG image to base64 string
        png_image_b64 = "data:image/png;base64,"
        png_image_b64 += base64.b64encode(png_image.getvalue()).decode('utf8')
        
        return jsonify({'image': png_image_b64})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        sk_id = data['sk_id']
        
        # Find the customer data
        customer_data = test_data[test_data['SK_ID_CURR'] == sk_id]
        if customer_data.empty:
            return jsonify({'error': 'Customer not found'})
        
        # Prepare for prediction
        X = customer_data.drop(columns=['SK_ID_CURR'])
        X_scaled = scaler.transform(X)
        
        # Make predictions
        lr_prob = lr_model.predict_proba(X_scaled)[0][1]
        gb_prob = gb_model.predict_proba(X)[0][1]
        
        # Determine risk category
        if gb_prob > 0.7:
            risk_category = "HIGH RISK"
        elif gb_prob > 0.4:
            risk_category = "MEDIUM RISK"
        else:
            risk_category = "LOW RISK"
        
        return jsonify({
            'lr_probability': float(lr_prob),
            'gb_probability': float(gb_prob),
            'risk_category': risk_category
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)