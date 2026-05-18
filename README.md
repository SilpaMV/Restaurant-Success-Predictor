# 🍽️ AI Restaurant Success Predictor

An end-to-end Machine Learning & AI-powered restaurant analytics web application that predicts restaurant ratings and business success potential using real-world restaurant data, advanced feature engineering, XGBoost regression modeling, and Generative AI business recommendations.

This project combines:
- 📊 Data Science & Predictive Analytics
- 🤖 Machine Learning Model Optimization
- 🧠 Generative AI Suggestions using GROQ API
- 🌐 Flask Web Deployment
- 🎨 Responsive Dashboard UI

---

# 🧠 Problem Statement

Restaurant business success depends on multiple factors such as:

- cuisine diversity
- online ordering
- delivery availability
- booking facilities
- city demographics
- tourism
- pricing
- operational strategy

This project aims to predict restaurant ratings intelligently using Machine Learning and provide actionable AI-generated recommendations for business improvement.

---

# 🌐 Data Collection

The dataset used for this project was collected through web scraping techniques from restaurant listing platforms including Zomato and other restaurant-related sources.

The collected data included:
- Restaurant names
- City information
- Cuisine types
- Cost for two
- Online ordering availability
- Table booking availability
- Delivery services
- Ratings
- Business-related restaurant features

This real-world dataset was further cleaned, preprocessed, and transformed for machine learning model training and analysis.

---

# 📂 Dataset Features

The model uses multiple engineered business features including:

| Feature | Description |
|---|---|
| city | Restaurant location |
| cuisine | Cuisine types |
| cuisine_count | Number of cuisines |
| cost_log | Restaurant cost factor |
| order_online | Online ordering availability |
| book_table | Table booking facility |
| delivery_available | Delivery support |
| dinein_available | Dine-in support |
| city_population | Population of city |
| city_area_km | City area |
| tourism_flag | Tourism influence |
| gdp_flag | GDP/business area indicator |

---

# 🎯 Key Features

✅ End-to-End Data Science Workflow  
✅ Advanced EDA & Visualization  
✅ Feature Engineering Pipeline  
✅ Multiple ML Models Tested & Tuned  
✅ Hyperparameter Optimization using GridSearchCV  
✅ Best Model Selection using XGBoost Regressor  
✅ Flask-Based Web Application  
✅ Modern Responsive Dashboard UI  
✅ AI Business Suggestions using GROQ API  
✅ Interactive Cuisine Selection System  
✅ Real-Time Prediction System  
✅ Recruiter-Friendly Project Architecture  

---


# ⚙️ Complete Machine Learning Workflow

## ✔️ Data Collection

- Restaurant business dataset collection
- Structured feature analysis

---

## 📌 Data Preprocessing
- Missing value handling
- Duplicate removal
- Feature cleaning
- Categorical encoding
- Numerical scaling
- Custom MultiLabelEncoder for cuisines

---

## ✔️ Feature Engineering

Advanced feature engineering techniques were implemented including:

- Cuisine count generation
- Cost transformation
- Multi-label cuisine encoding
- Business indicator creation
- Demographic feature processing

---

# 🤖 Machine Learning Model Experimentation

Multiple Machine Learning algorithms were trained, tuned and evaluated to identify the best estimator for restaurant success prediction.

---

## 📊 Models Evaluated

Multiple regression models were trained and evaluated:

| Model | Status |
|------|------|
| Linear Regression | Tested |
| Decision Tree Regressor | Tested |
| Random Forest Regressor | Tested |
| Gradient Boosting Regressor | Tested |
| KNN Regressor | Tested |
| SVR | Tested |
| XGBoost Regressor | ✅ Final Selected Model |

---

# 🔧 Hyperparameter Tuning

All machine learning models were systematically tuned using hyperparameter optimization techniques to improve:

- prediction accuracy
- model generalization
- robustness
- cross-validation performance

Techniques used:

- GridSearchCV
- Cross Validation
- Performance Comparison
- Estimator Optimization

---

# 📈 Evaluation Metrics

The models were evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Cross Validation Scores

After comparative analysis and tuning, the XGBoost Regressor achieved the best overall performance and was selected as the final production model.

---

# 🏆 Final Model

## ✅ Tuned XGBoost Regressor

Why XGBoost?
- Better generalization
- High prediction accuracy
- Handles feature interactions efficiently
- Strong performance on structured/tabular datasets
- Robust against overfitting

---

# 🌐 Flask Web Application

The trained ML model was deployed using Flask with a fully responsive AI dashboard interface.

## Features Included
- Responsive Design
- Animated Components
- Interactive Cuisine Input
- Dynamic Prediction Results
- AI Suggestion Engine
- Real-Time User Inputs
- Professional Dashboard Styling

---

# 🤖 Generative AI Integration

Integrated GROQ API for intelligent restaurant improvement suggestions.

If predicted rating is below **4.2**, the system generates:
- Business improvement recommendations
- Marketing suggestions
- Customer experience enhancements
- Revenue optimization ideas

Fallback logic is also implemented in case of API interruption.

---


# 🛠️ Technologies Used

## Programming & ML
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

## Visualization
- Matplotlib
- Seaborn

## Deployment
- Flask
- HTML
- CSS
- JavaScript

## AI Integration
- GROQ API
- Llama 3 Model

---

# 📂 Project Structure

```bash
Restaurant-Success-Predictor/
│
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
├── Restaurant_Success_Predictor.ipynb
│
├── templates/
│   └── index.html
│
├── screenshots/
│   ├── homepage.png
│   └── result.png
│
└── demo/
    └── demo_video.mp4
```
---

# 📸 Project Screenshots

## 🏠 Home Dashboard
(Add your screenshot here)

```md
![Home Page](screenshots/homepage.png)
```

---

## 📈 Prediction Result
(Add your result screenshot here)

```md
![Prediction Result](screenshots/result.png)
```

---

# 🎥 Project Demo Video

Upload your demo video inside GitHub repository and use:

```md
[▶ Watch Demo Video](demo/demo_video.mp4)
```

---

# 📓 Jupyter Notebook

The repository also contains the complete Jupyter/Colab notebook including:

- EDA
- preprocessing
- feature engineering
- model training
- model tuning
- evaluation
- estimator comparison

---

# 📈 Business Impact

This project demonstrates how AI and Machine Learning can help:

- predict restaurant performance
- improve operational strategy
- assist business decision making
- generate intelligent business insights

---

# 👩‍💻 Developed By

## Silpa M V

Electronics & Communication Engineer | AI & Data Science Enthusiast | Machine Learning Developer | Flask Application Developer

Passionate about building real-world AI solutions using Machine Learning, Predictive Analytics and Intelligent Web Applications.

---
