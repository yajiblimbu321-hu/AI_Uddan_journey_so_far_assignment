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
* Objective: Predict binary student dropout risk (DropoutRisk: 0 for low risk, 1 for high risk) using multiple academic and behavioral metrics[cite: 1].
* Dataset: datasets/Question1.csv[cite: 1].
* Features: Attendance, AssignmentCompletion, PreviousMarks, MissedClasses, LateSubmissions[cite: 1].
* Workflow:
  * Inspected data distribution, validated absence of null values, and generated summary statistics[cite: 1].
  * Conducted an 80/20 train-test split using stratify=Y to preserve class ratios[cite: 1].
  * Standardized feature scales using StandardScaler[cite: 1].
  * Trained a LogisticRegression model, achieving 100% accuracy, precision, recall, and F1-score on the test split[cite: 1].
  * Extracted feature coefficients (model.coef_) to analyze directional influence:
    * Negative weights: Attendance (-0.748) and AssignmentCompletion (-0.655) reduce dropout risk[cite: 1].
    * Positive weights: MissedClasses (+0.745) and LateSubmissions (+0.686) increase dropout risk[cite: 1].
  * Performed inference on sample student profiles using predict_proba()[cite: 1].

### 2. multila.ipynb — Multi-Label Intervention Classification
* Objective: Simultaneously predict multiple non-exclusive student support requirements across three distinct target tracks: MathSupport, EnglishSupport, and AttendanceSupport[cite: 2].
* Dataset: datasets/Question2.csv[cite: 2].
* Features: Attendance, MathScore, EnglishScore, QuizScore, MissedClasses[cite: 2].
* Workflow:
  * Defined a 2D binary label matrix Y for multi-label learning[cite: 2].
  * Configured OneVsRestClassifier(LogisticRegression()) to fit independent binary estimators for each support category[cite: 2].
  * Evaluated joint prediction arrays and prediction probabilities on the holdout test set[cite: 2].
  * Implemented an inference pipeline that accepts raw student metrics and prints formatted support recommendations (YES/NO per subject)[cite: 2].

### 3. multipleCLASS.ipynb — Multivariable Logistic Regression with Cross-Validation
* Objective: Classify student exam results (Result: Pass or Fail) using multivariable logistic regression optimized through automated cross-validation[cite: 3].
* Dataset: datasets/student_multiple_binary_classification.csv[cite: 3].
* Features: StudyHours, Attendance, PreviousMarks[cite: 3].
* Workflow:
  * Encoded categorical target values (Pass -> 1, Fail -> 0)[cite: 3].
  * Trained a LogisticRegressionCV() model to automatically validate and select the optimal inverse regularization parameter C over stratified folds[cite: 3].
  * Verified model predictions against actual test labels, achieving a 1.0 accuracy score[cite: 3].

### 4. assig6.ipynb — Multiple Linear Regression & Data Preprocessing
* Objective: Predict continuous final exam performance (FinalMarks) based on multiple academic factors[cite: 4].
* Dataset: datasets/student_multiple_features.csv[cite: 4].
* Features: StudyHours, Attendance, PreviousMarks[cite: 4].
* Workflow:
  * Identified data entry anomalies, including missing cells (NaN) and non-numeric strings ('Absent')[cite: 4].
  * Coerced dirty columns using pd.to_numeric(errors="coerce") and imputed missing values with median statistics[cite: 4].
  * Fitted a LinearRegression() model and extracted parameters[cite: 4]:
    * Model Intercept: -24.73[cite: 4].
    * Weights: StudyHours (0.470), Attendance (0.547), PreviousMarks (0.740)[cite: 4].
  * Generated predicted final grades for multiple candidate student records[cite: 4].

### 5. assign3.ipynb — LMS Data Cleaning, Anomaly Handling & Outlier Detection
* Objective: Clean, standardize, and treat outliers on a raw, inconsistent Learning Management System (LMS) dataset[cite: 5].
* Dataset: datasets/LMS_Dirty_Dataset.csv[cite: 5].
* Workflow:
  * Standardized text columns by trimming whitespace and applying title casing (.str.title())[cite: 5].
  * Removed string symbols (e.g., stripping % signs) and cast columns to appropriate numerical and datetime types[cite: 5].
  * Corrected invalid negative values across strictly positive columns (course_fee, assignment_score) using .abs()[cite: 5].
  * Evaluated and compared three outlier treatment strategies:
    * IQR Filtering: Identified extreme deviations using [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR] fences[cite: 5].
    * Clipping & Imputation: Applied .clip() to constrain outliers within threshold boundaries and tested median substitution[cite: 5].
    * Z-Score Normalization: Calculated parametric z-scores to flag observations with |z| > 3[cite: 5].
  * Performed course-level aggregations to analyze fee ranges and maximum scholarship discount rates[cite: 5].

### 6. assign4.ipynb — Exploratory Data Analysis & Time-Series Visualizations
* Objective: Analyze longitudinal student enrollment trends, course progression, attendance, and revenue metrics over time[cite: 6].
* Dataset: datasets/dp_lms_student_analytics_dataset.csv[cite: 6].
* Workflow:
  * Parsed date fields and extracted monthly periods using dt.to_period("M")[cite: 6].
  * Aggregated temporal metrics: monthly student enrollment volume, monthly revenue sums, and average completion rates[cite: 6].
  * Created data visualizations using matplotlib.pyplot:
    * Monthly Enrollment Trend Line Chart[cite: 6].
    * Monthly Total Revenue Trajectory[cite: 6].
    * Average Monthly Course Progress and Attendance Progression[cite: 6].
    * Course-wise Student Distribution Bar Chart[cite: 6].

### 7. assign6.ipynb — Simple Linear Regression
* Objective: Model the direct relationship between daily study duration and final exam marks[cite: 7].
* Dataset: datasets/student_marks.csv[cite: 7].
* Features: StudyHours (reshaped to a 2D array for Scikit-Learn compatibility)[cite: 7].
* Target: FinalMarks[cite: 7].
* Workflow:
  * Split data into training and testing sets (80/20 ratio)[cite: 7].
  * Fitted a LinearRegression() model to derive the best-fit line: FinalMarks = 9.25 * StudyHours + 17.74[cite: 7].
  * Evaluated score predictions across specific study hour intervals (2, 4, and 6 hours)[cite: 7].

### 8. classification.ipynb — Univariate Logistic Regression
* Objective: Classify student exam outcomes (Pass vs Fail) based solely on study hours[cite: 8].
* Dataset: datasets/student_binary_classification.csv[cite: 8].
* Features & Target: StudyHours -> Result (Pass -> 1, Fail -> 0)[cite: 8].
* Workflow:
  * Mapped string labels to binary integers and formatted the input feature into a 2D array[cite: 8].
  * Trained a LogisticRegression() model on the training split[cite: 8].
  * Validated predictions against ground truth, obtaining 100% test accuracy[cite: 8].
  * Evaluated the sigmoid decision boundary across boundary test points (2.0, 4.5, and 5.0 hours)[cite: 8].

### 9. knnex.ipynb — K-Nearest Neighbors (KNN) & Historical Neighbor Tracing
* Objective: Classify student pass/fail status using instance-based learning and retrieve the most similar historical student profiles[cite: 9].
* Dataset: datasets/Question3.csv[cite: 9].
* Features: Attendance, PreviousMarks, AssignmentCompletion, QuizScore[cite: 9].
* Target: Pass (0 or 1)[cite: 9].
* Workflow:
  * Normalized all feature dimensions using StandardScaler to ensure uniform Euclidean distance weighting[cite: 9].
  * Trained a KNeighborsClassifier(n_neighbors=3) model, achieving 1.0 accuracy, precision, recall, and F1-score[cite: 9].
  * Used model.kneighbors() on new student data to extract neighbor distances and internal training set indices[cite: 9].
  * Implemented an index-mapping step to link internal array indices back to original DataFrame row labels, identifying the exact records of the 3 most similar historical students[cite: 9].

