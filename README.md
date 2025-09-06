# Home Credit Indonesia - Loan Default Prediction

![Dashboard Preview](https://i.postimg.cc/KcsQNJLQ/807dcabc2ea5f525d928466f98ed5b95-1655085458-293-removebg-preview.png)

A comprehensive machine learning solution for predicting loan default risk, built with Python and Flask. This project helps financial institutions make better lending decisions by identifying customers with repayment capacity while minimizing risk. This project is made as the final task for my internship with Home Credit Indonesia.

## 📊 Project Overview

This project addresses the critical challenge of loan default prediction in the financial industry. By leveraging machine learning models, we can:

- ✅ **Prevent good customers from being rejected**
- ✅ **Optimize loan terms** (principal, maturity, repayment calendars)
- ✅ **Identify high-risk applicants** early in the process
- ✅ **Provide transparent, explainable decisions** to stakeholders

## 🚀 Features

### Machine Learning Models
- **Logistic Regression** (AUC: 0.84) - For interpretability and regulatory compliance
- **Gradient Boosting** (AUC: 0.97) - For superior predictive performance
- **Advanced Feature Engineering** from multiple data sources
- **Class Imbalance Handling** using SMOTE and class weights

### Interactive Dashboard
- **Real-time predictions** for individual applicants
- **Comprehensive visualizations**:
  - Target distribution analysis
  - Feature correlation heatmaps
  - Income vs credit scatter plots
  - ROC curve comparisons
  - Feature importance charts
- **Responsive design** with modern cyberpunk aesthetic
- **Risk categorization** with color-coded indicators

### Data Processing
- **Multi-source data integration** from 7 different CSV files
- **Advanced feature engineering** with bureau data aggregation
- **Missing value handling** with intelligent imputation
- **Outlier detection and treatment**

## 📁 Project Structure

```
home-credit-default-risk/
├── app.py                      # Flask application
├── requirements.txt            # Python dependencies
├── notebook-1.ipynb           # Data preprocessing & EDA
├── notebook-2.ipynb           # Model training & evaluation
├── templates/
│   └── index.html             # Dashboard interface
├── models/                    # Trained model files
│   ├── logistic_regression_model.pkl
│   ├── gradient_boosting_model.pkl
│   └── scaler.pkl
├── processed_data/            # Processed datasets
│   ├── processed_train.csv
│   └── processed_test.csv
└── README.md
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/home-credit-default-risk.git
   cd home-credit-default-risk
   ```

2. **Create virtual environment**
   ```bash
   python -m venv hcienv
   source hcienv/bin/activate  # Linux/Mac
   # or
   hcienv\Scripts\activate    # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the dashboard**
   Open your browser and navigate to `http://localhost:5000`

## 📈 Data Sources

**Note: Due to NDA restrictions, actual data files are not included in this repository.**

The project utilizes the following data sources (structure only):

- `application_{train|test}.csv` - Main application data
- `bureau.csv` - Previous credits from other institutions
- `bureau_balance.csv` - Monthly credit balances
- `POS_CASH_balance.csv` - Point of sales and cash loans
- `credit_card_balance.csv` - Credit card balance history
- `previous_application.csv` - Previous Home Credit applications
- `installments_payments.csv` - Repayment history

## 🔧 Usage

### 1. Data Preprocessing
Run `notebook-1.ipynb` to:
- Clean and preprocess raw data
- Perform feature engineering
- Generate exploratory visualizations
- Create processed datasets for modeling

### 2. Model Training
Run `notebook-2.ipynb` to:
- Train Logistic Regression and Gradient Boosting models
- Evaluate model performance
- Generate feature importance charts
- Save trained models for deployment

### 3. Dashboard Usage
Access the web dashboard to:
- View data insights and visualizations
- Make individual predictions by entering `SK_ID_CURR`
- Compare model performances
- Understand feature importance

## 🎯 Model Performance

| Model | AUC Score | Key Strengths |
|-------|-----------|---------------|
| Logistic Regression | 0.84 | Interpretability, Regulatory compliance |
| Gradient Boosting | 0.97 | Predictive accuracy, Non-linear patterns |

**Key Findings:**
- Top predictive features include credit history, income ratios, and external scores
- Class imbalance successfully handled with SMOTE
- Feature engineering significantly improved model performance

## 🏆 Business Impact

### Expected Outcomes
- **85-90% reduction** in bad loans
- **15-20% increase** in safe approvals
- **20-30% improvement** in risk-based pricing
- **Real-time decision making** capability

### Risk Categories
- **LOW RISK** (<30% probability): Automatic approval
- **MEDIUM RISK** (30-70%): Manual review recommended
- **HIGH RISK** (>70%): Decline with explanation

## 🔮 Future Enhancements

- [ ] Real-time API integration with loan systems
- [ ] Advanced model explainability with SHAP values
- [ ] Automated retraining pipeline
- [ ] Multi-language support
- [ ] Mobile-responsive design
- [ ] Advanced filtering and segmentation
- [ ] Export functionality for reports

## 🤝 Contributing

Due to the sensitive nature of financial data (CSVs) and NDA restrictions, contributions are currently limited. However, we welcome:

- Bug reports and feature suggestions
- Documentation improvements
- UI/UX enhancements
- Performance optimizations

## 📝 License

This project is proprietary and contains confidential information. All rights reserved.

## 📞 Support

For technical support or questions about implementation:
- Create an issue in this repository
- Contact the developer at michaelwenas@programmer.net

## 🎨 Acknowledgments

- Home Credit Indonesia for the problem statement
- Machine learning community for open-source libraries
- UI design inspired by modern cyberpunk aesthetics

---

**Disclaimer**: This project is for demonstration purposes. Actual implementation in production environments should follow appropriate regulatory guidelines and compliance requirements.
