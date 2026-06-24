# 🏠 House Price Prediction & Trend Analysis

A Machine Learning web application that predicts property prices and forecasts future price trends based on property features such as area, bedrooms, locality score, amenities, and property type.

## 🚀 Live Demo

[https://housepriceproject-k880.onrender.com]

---

## 📌 Features

- Predicts estimated property prices
- Forecasts future market trend (Up / Down)
- Supports multiple property types:
  - Villa
  - House
  - Flat
  - Apartment
  - Building
  - Farm House
  - Empty Land
- Interactive and user-friendly interface
- Real-time predictions using trained Machine Learning models

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries
- Pandas
- Scikit-learn
- Joblib
- Streamlit

### Machine Learning
- Regression Model for Price Prediction
- Classification Model for Trend Prediction

### Deployment
- Render

---

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── app.py
├── house_price_model.pkl
├── house_trend_model.pkl
├── requirements.txt
├── Procfile
├── house_data.csv
├── house_data_with_floors.csv
├── model_train.py
├── model_test.py
└── README.md
```

## 📊 Input Parameters

The application accepts the following inputs:

- Area (sqft)
- Bedrooms
- Bathrooms
- Swimming Pool
- Parking Spaces
- Furnished Status
- Furniture Availability
- Locality Score
- Access to Main Road
- Shopping Complex Nearby
- Local Market Nearby
- Previous Property Price
- Property Type
- Residence Status
- Registration Status
- Number of Floors

---

## 🎯 Outputs

### Price Prediction

The model estimates the expected property price based on the provided features.

### Trend Prediction

The model predicts whether the property price trend is likely to:

- ⬆️ Increase (UP)
- ⬇️ Decrease (DOWN)

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/house-price-prediction.git
```

Move to project folder:

```bash
cd house-price-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📈 Future Improvements

- Property price visualizations
- Location-based predictions using maps
- Historical trend charts
- Model performance dashboard
- API integration for real estate listings

---

## 👨‍💻 Author

Peeyush Choudhary

📧 peeyush142186@gmail.com

📱 +91 7678489150

GitHub:
https://github.com/peeyush-coder


---

## ⭐ Project Objective

This project was developed to demonstrate practical skills in:

- Data Analysis
- Data Preprocessing
- Machine Learning
- Model Deployment
- Streamlit Application Development
- End-to-End Data Science Workflow

Suitable for showcasing in Data Analyst, Business Analyst, and Data Science portfolios.
