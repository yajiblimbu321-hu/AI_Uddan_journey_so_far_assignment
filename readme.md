# Machine Learning & Data Analytics Portfolio — AI Uddan

A comprehensive repository of end-to-end Machine Learning implementations, exploratory data analysis, data preprocessing, and predictive modeling developed during the AI Uddan training program.

## Repository Structure

    datasets/
    │   ├── Question1.csv
    │   ├── Question2.csv
    │   ├── Question3.csv
    │   ├── LMS_Dirty_Dataset.csv
    │   ├── dp_lms_student_analytics_dataset.csv
    │   ├── student_marks.csv
    │   ├── student_multiple_features.csv
    │   ├── student_binary_classification.csv
    │   └── student_multiple_binary_classification.csv
    ├── logisticone.ipynb
    ├── multila.ipynb
    ├── multipleCLASS.ipynb
    ├── assig6.ipynb
    ├── assign3.ipynb
    ├── assign4.ipynb
    ├── assign6.ipynb
    ├── classification.ipynb
    ├── knnex.ipynb
    └── README.md

## Notebooks & Implementation Details

### 1. logisticone.ipynb — Binary Logistic Regression & Feature Interpretation
* Objective: Predict binary student dropout risk (DropoutRisk: 0 for low risk, 1 for high risk) using multiple academic and behavioral metrics.
* Dataset: datasets/Question1.csv
* Features: Attendance, AssignmentCompletion, PreviousMarks, MissedClasses, LateSubmissions
* Workflow:
  * Inspected data distribution, validated absence of null values, and generated summary statistics.
  * Conducted an 80/20 train-test split using stratify=Y to preserve class ratios.
  * Standardized feature scales using StandardScaler.
  * Trained a LogisticRegression model, achieving 100% accuracy, precision, recall, and F1-score on the test split.
  * Extracted feature coefficients (model.coef_) to analyze directional influence:
    * Negative weights: Attendance (-0.748) and AssignmentCompletion (-0.655) reduce dropout risk.
    * Positive weights: MissedClasses (+0.745) and LateSubmissions (+0.686) increase dropout risk.
  * Performed inference on sample student profiles using predict_proba().

### 2. multila.ipynb — Multi-Label Intervention Classification
* Objective: Simultaneously predict multiple non-exclusive student support requirements across three distinct target tracks: MathSupport, EnglishSupport, and AttendanceSupport.
* Dataset: datasets/Question2.csv
* Features: Attendance, MathScore, EnglishScore, QuizScore, MissedClasses
* Workflow:
  * Defined a 2D binary label matrix Y for multi-label learning.
  * Configured OneVsRestClassifier(LogisticRegression()) to fit independent binary estimators for each support category.
  * Evaluated joint prediction arrays and prediction probabilities on the holdout test set.
  * Implemented an inference pipeline that accepts raw student metrics and prints formatted support recommendations (YES/NO per subject).

### 3. multipleCLASS.ipynb — Multivariable Logistic Regression with Cross-Validation
* Objective: Classify student exam results (Result: Pass or Fail) using multivariable logistic regression optimized through automated cross-validation.
* Dataset: datasets/student_multiple_binary_classification.csv
* Features: StudyHours, Attendance, PreviousMarks
* Workflow:
  * Encoded categorical target values (Pass -> 1, Fail -> 0).
  * Trained a LogisticRegressionCV() model to automatically validate and select the optimal inverse regularization parameter C over stratified folds.
  * Verified model predictions against actual test labels, achieving a 1.0 accuracy score.

### 4. assig6.ipynb — Multiple Linear Regression & Data Preprocessing
* Objective: Predict continuous final exam performance (FinalMarks) based on multiple academic factors.
* Dataset: datasets/student_multiple_features.csv
* Features: StudyHours, Attendance, PreviousMarks
* Workflow:
  * Identified data entry anomalies, including missing cells (NaN) and non-numeric strings ('Absent').
  * Coerced dirty columns using pd.to_numeric(errors="coerce") and imputed missing values with median statistics.
  * Fitted a LinearRegression() model and extracted parameters:
    * Model Intercept: -24.73
    * Weights: StudyHours (0.470), Attendance (0.547), PreviousMarks (0.740).
  * Generated predicted final grades for multiple candidate student records.

### 5. assign3.ipynb — LMS Data Cleaning, Anomaly Handling & Outlier Detection
* Objective: Clean, standardize, and treat outliers on a raw, inconsistent Learning Management System (LMS) dataset.
* Dataset: datasets/LMS_Dirty_Dataset.csv
* Workflow:
  * Standardized text columns by trimming whitespace and applying title casing (.str.title()).
  * Removed string symbols (e.g., stripping % signs) and cast columns to appropriate numerical and datetime types.
  * Corrected invalid negative values across strictly positive columns (course_fee, assignment_score) using .abs().
  * Evaluated and compared three outlier treatment strategies:
    * IQR Filtering: Identified extreme deviations using [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR] fences.
    * Clipping & Imputation: Applied .clip() to constrain outliers within threshold boundaries and tested median substitution.
    * Z-Score Normalization: Calculated parametric z-scores to flag observations with |z| > 3.
  * Performed course-level aggregations to analyze fee ranges and maximum scholarship discount rates.

### 6. assign4.ipynb — Exploratory Data Analysis & Time-Series Visualizations
* Objective: Analyze longitudinal student enrollment trends, course progression, attendance, and revenue metrics over time.
* Dataset: datasets/dp_lms_student_analytics_dataset.csv
* Workflow:
  * Parsed date fields and extracted monthly periods using dt.to_period("M").
  * Aggregated temporal metrics: monthly student enrollment volume, monthly revenue sums, and average completion rates.
  * Created data visualizations using matplotlib.pyplot:
    * Monthly Enrollment Trend Line Chart.
    * Monthly Total Revenue Trajectory.
    * Average Monthly Course Progress and Attendance Progression.
    * Course-wise Student Distribution Bar Chart.

### 7. assign6.ipynb — Simple Linear Regression
* Objective: Model the direct relationship between daily study duration and final exam marks.
* Dataset: datasets/student_marks.csv
* Features: StudyHours (reshaped to a 2D array for Scikit-Learn compatibility)
* Target: FinalMarks
* Workflow:
  * Split data into training and testing sets (80/20 ratio).
  * Fitted a LinearRegression() model to derive the best-fit line: FinalMarks = 9.25 * StudyHours + 17.74.
  * Evaluated score predictions across specific study hour intervals (2, 4, and 6 hours).

### 8. classification.ipynb — Univariate Logistic Regression
* Objective: Classify student exam outcomes (Pass vs Fail) based solely on study hours.
* Dataset: datasets/student_binary_classification.csv
* Features & Target: StudyHours -> Result (Pass -> 1, Fail -> 0)
* Workflow:
  * Mapped string labels to binary integers and formatted the input feature into a 2D array.
  * Trained a LogisticRegression() model on the training split.
  * Validated predictions against ground truth, obtaining 100% test accuracy.
  * Evaluated the sigmoid decision boundary across boundary test points (2.0, 4.5, and 5.0 hours).

### 9. knnex.ipynb — K-Nearest Neighbors (KNN) & Historical Neighbor Tracing
* Objective: Classify student pass/fail status using instance-based learning and retrieve the most similar historical student profiles.
* Dataset: datasets/Question3.csv
* Features: Attendance, PreviousMarks, AssignmentCompletion, QuizScore
* Target: Pass (0 or 1)
* Workflow:
  * Normalized all feature dimensions using StandardScaler to ensure uniform Euclidean distance weighting.
  * Trained a KNeighborsClassifier(n_neighbors=3) model, achieving 1.0 accuracy, precision, recall, and F1-score.
  * Used model.kneighbors() on new student data to extract neighbor distances and internal training set indices.
  * Implemented an index-mapping step to link internal array indices back to original DataFrame row labels, identifying the exact records of the 3 most similar historical students.

