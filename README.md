# 🛒 Flipkart Product Performance & Pricing Analysis

> **AICTE IBM SkillsBuild Internship Project**
> An end-to-end data analytics and machine learning project for analyzing Flipkart product reviews, pricing patterns, customer sentiment, product performance, and factors associated with product ratings.

---

## 📌 Project Overview

This project analyzes Flipkart product review data using **Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and Streamlit**.

The project combines:

* Data cleaning and preprocessing
* Exploratory data analysis
* Product performance analysis
* Pricing analysis
* Customer sentiment analysis
* Machine learning
* TF-IDF based text analysis
* Interactive Streamlit visualization
* Business-oriented recommendations

### 🎯 Key Business Questions

| Business Question                                    | Approach                                   |
| ---------------------------------------------------- | ------------------------------------------ |
| Which products perform well and which underperform?  | Composite product-performance score        |
| How does pricing vary across product categories?     | Price distribution and price-tier analysis |
| Which variables are associated with product ratings? | Correlation analysis and Gradient Boosting |
| What are the major customer complaints?              | TF-IDF analysis of negative reviews        |
| How can the findings support business decisions?     | Data-driven recommendations                |

---

## 🔍 Analysis Pipeline

The project follows an end-to-end analytics workflow:

```text
Raw Review Data
      ↓
Data Cleaning & Preprocessing
      ↓
Category Extraction
      ↓
Product-Level Aggregation
      ↓
Exploratory Data Analysis
      ↓
Pricing & Sentiment Analysis
      ↓
Machine Learning
      ↓
Negative Review / TF-IDF Analysis
      ↓
Business Insights & Recommendations
      ↓
Interactive Streamlit Dashboard
```

---

## 📊 Dataset

The analysis uses a **Flipkart customer review dataset obtained from Kaggle**.

The dataset is **not included in this repository**.

This was intentionally excluded to keep the repository lightweight and to avoid redistributing the raw dataset.

### Dataset contains information such as:

* Product name
* Product price
* Customer rating
* Review text
* Review summary
* Customer sentiment

### Dataset Source

The dataset used for this project is available through Kaggle:

**Flipkart Product Customer Reviews dataset** by `niraliivaghani`.

> To run the complete analysis locally, download the dataset separately and place the CSV file in the project directory using the filename expected by the Python application.

---

## 📁 Project Structure

```text
Flipkart-Product-Performance-Pricing-Analysis/
│
├── README.md
├── requirements.txt
├── Samistha_flipkartproductperformance_pricinganalysis.py
└── Samistha_ProjectReport.docx
```

### File Description

| File                                                     | Description                                                                             |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `Samistha_flipkartproductperformance_pricinganalysis.py` | Main Python analysis pipeline and Streamlit application                                 |
| `requirements.txt`                                       | Python dependencies required to run the project                                         |
| `Samistha_ProjectReport.docx`                            | Detailed project report containing methodology, analysis, findings, and recommendations |
| `README.md`                                              | Project documentation                                                                   |

> **Note:** The dataset and generated chart images are intentionally not included in the repository.

---

## ⚙️ Technologies Used

| Technology       | Purpose                                               |
| ---------------- | ----------------------------------------------------- |
| **Python**       | Core programming language                             |
| **Pandas**       | Data manipulation and analysis                        |
| **NumPy**        | Numerical computation                                 |
| **Matplotlib**   | Data visualization                                    |
| **Seaborn**      | Statistical visualization                             |
| **Scikit-learn** | Machine learning and text analysis                    |
| **Streamlit**    | Interactive web dashboard                             |
| **WordCloud**    | Visualization of frequently occurring complaint terms |

---

## 🧹 Data Preprocessing

The preprocessing stage includes:

* Standardizing column names
* Cleaning product prices
* Handling missing values
* Removing duplicate records
* Normalizing sentiment labels
* Converting ratings into numerical values
* Preparing review text for analysis
* Extracting product categories from product names

Product categories are inferred using a **rule-based keyword mapping** from product names.

---

## 📈 Product Performance Analysis

Products are aggregated using metrics such as:

* Average rating
* Number of reviews
* Average price
* Positive sentiment percentage
* Negative sentiment percentage

A composite performance score is then used to identify relatively higher- and lower-performing products.

The analysis helps identify patterns in:

* Customer satisfaction
* Review engagement
* Product performance
* Category-level performance

---

## 💰 Pricing Analysis

The project examines relationships between product pricing and customer feedback.

Products are grouped into price tiers based on their distribution:

* **Budget**
* **Mid-range**
* **Premium**

The analysis compares these groups using:

* Average rating
* Sentiment distribution
* Price ranges
* Category-level pricing patterns

---

## 🤖 Machine Learning Analysis

A **Gradient Boosting Regressor** is used to examine which available product-level variables are most strongly associated with average product ratings.

The analysis includes:

* Train/test split
* Feature preparation
* Model training
* Prediction
* MAE evaluation
* R² evaluation
* Feature importance analysis

### Important Interpretation

Feature importance indicates **association/predictive usefulness**, not causation.

For example, sentiment-related variables may have a strong relationship with ratings because both are derived from customer feedback. Therefore, the model results should be interpreted as predictive associations rather than proof that a particular variable directly causes higher ratings.

---

## 📝 Customer Complaint Analysis

Negative customer reviews are analyzed using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

This helps identify frequently occurring terms and phrases associated with negative customer experiences.

The analysis can highlight potential complaint areas such as:

* Product quality
* Packaging
* Product description mismatch
* Size-related issues
* Delivery/refund-related concerns

These findings are used to generate business-oriented recommendations.

---

## 💡 Business Recommendations

The project translates analytical findings into practical recommendations related to:

* Product quality
* Pricing strategy
* Customer satisfaction
* Product descriptions
* Packaging
* Negative-review patterns
* Category-level performance

The recommendations are based on patterns observed in the analyzed dataset and should be interpreted within the limitations of the available data.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit application that allows users to explore the analysis.

The dashboard provides functionality such as:

* Product performance exploration
* Category filtering
* Minimum review filtering
* Top/bottom product analysis
* Pricing analysis
* Rating and sentiment analysis
* Machine learning insights
* Negative-review analysis

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/SamisthaKesarwani/Flipkart-Product-Performance-Pricing-Analysis.git
cd Flipkart-Product-Performance-Pricing-Analysis
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the dataset separately and place the CSV file in the project directory using the filename expected by the application.

---

## ▶️ Running the Application

Launch the Streamlit dashboard with:

```bash
streamlit run Samistha_flipkartproductperformance_pricinganalysis.py
```

The application will open in your browser.

You can then explore the analysis interactively using the dashboard controls.

---

## 📋 Analysis Sections

The project contains the following major stages:

1. **Data Loading**
2. **Data Cleaning**
3. **Category Extraction**
4. **Product-Level Aggregation**
5. **Product Performance Analysis**
6. **Pricing Pattern Analysis**
7. **Rating Association Analysis**
8. **Negative Review Analysis**
9. **Business Recommendations**
10. **Interactive Streamlit Dashboard**

---

## 📄 Project Report

A detailed project report is included in the repository:

`Samistha_ProjectReport.docx`

The report covers:

* Business problem
* Objectives
* Dataset description
* Data preprocessing
* Exploratory analysis
* Machine learning methodology
* Sentiment analysis
* Key findings
* Recommendations
* Limitations
* Conclusion

---

## ⚠️ Limitations

The analysis has several limitations:

* The dataset represents a specific collection of Flipkart reviews and may not represent all Flipkart products or customers.
* Product categories are inferred using keyword-based rules.
* Sentiment labels are based on the labels available in the dataset.
* Correlation and machine-learning feature importance should not be interpreted as causal relationships.
* The analysis is based on historical/static review data rather than continuously updated marketplace data.
* Business recommendations depend on the quality and coverage of the underlying dataset.

---

## 🎓 Internship Project

**AICTE–IBM SkillsBuild Internship**

### Project Title

**Flipkart Product Performance & Pricing Analysis**

### Developed With

**IBM Bob | Python | Data Analysis | Machine Learning | NLP | Streamlit**

This project was developed with the assistance of **IBM Bob**, an AI-powered development tool, for implementing and refining the data analysis, machine learning, visualization, and Streamlit components.

---

## 👩‍💻 Author

**Samistha Kesarwani**

B.Tech Student

---

⭐ If you find this project useful, feel free to explore the repository and review the methodology and implementation.
