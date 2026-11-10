# Automated Cardiovascular Risk Prediction Through Multi-Algorithm Benchmarking and Containerized Machine Learning Operations

**Author One** (*Computer Science Dept.*, `author1@institution.edu`) | **Author Two** (*Computer Science Dept.*, `author2@institution.edu`) | **Author Three** (*Computer Science Dept.*, `author3@institution.edu`)

***Abstract - Early detection of coronary artery disease reduces mortality by enabling timely clinical intervention, yet deploying diagnostic models reliably across heterogeneous hospital cohorts remains challenging. Patient records frequently contain substantial missing values, inconsistent feature formatting, and high sensitivity requirements where missed positive cases carry severe clinical consequences. This paper presents an end-to-end clinical risk prediction framework and containerized machine learning operations pipeline built on 920 multi-center diagnostic records spanning four international medical institutions. To eliminate data leakage and accommodate missing diagnostic tests affecting up to 66.4 percent of records, a unified preprocessing pipeline performs stratified splitting before median numeric imputation, standard scaling, and one-hot encoding. Five supervised classifiers were benchmarked and tuned through five-fold cross-validated grid search. Tuned Random Forest achieved 0.8696 accuracy, 0.8611 precision, 0.9118 recall, 0.8857 harmonic mean score, and 0.9180 area under the receiver operating characteristic curve on 184 held-out patients, producing only nine false negatives. Experiments and deterministic reproducibility runs were logged in MLflow under a 0.75 minimum recall gate, and the production model was packaged into a containerized web service with physiological boundary validation, three-tier risk stratification, and automated endpoint testing passing eight of eight checks.***

***Index Terms - Cardiovascular risk, clinical decision support, containerization, machine learning operations.***


## Introduction

Cardiovascular diseases remain the leading cause of global mortality, making timely screening and risk stratification essential for preventive clinical workflows. Machine learning classifiers trained on non-invasive demographic, electrocardiographic, and stress-test measurements offer rapid decision support prior to invasive angiography [1]. However, translating an exploratory clinical model into a dependable hospital-ready service presents substantial engineering obstacles.

Multi-institutional diagnostic records exhibit severe missingness in specialized test columns such as fluoroscopy vessel counts and thalassemia assessments, creating strong risks of preprocessing leakage if statistics are computed prior to train-test splitting. Moreover, clinical microservices must enforce strict physiological input validation, maintain deterministic reproducibility across training runs, prioritize diagnostic recall to minimize missed disease cases, and operate consistently inside portable containers [2], [3]. This study addresses these gaps by uniting five-algorithm comparative modeling with a modular Machine Learning Operations (MLOps) lifecycle pipeline and a Dockerized FastAPI clinical inference service.


## Clinical Dataset and Preprocessing Pipeline


### I. Multi-Center UCI Cohort and Missingness Characterization

Experiments utilize the multi-center UCI Heart Disease dataset containing 920 patient records and 16 original columns collected across four clinical sites: Cleveland (304 records), Hungary (293 records), VA Long Beach (200 records), and Switzerland (123 records) [4]. Patient ages span 28 to 77 years with a mean age of 53.51 years, comprising 726 male and 194 female patients. The original angiographic severity outcome ranges from 0 to 4 (411 healthy records, and 265, 109, 107, and 28 records across grades 1 through 4). For clinical screening, the target is binarized such that any positive severity grade indicates heart disease, yielding 509 positive cases (55.33%) and 411 negative cases (44.67%).

Non-predictive identifier and hospital-origin columns are removed prior to modeling to prevent site confounding, leaving 13 clinical predictors: six numerical measurements and seven categorical or boolean attributes. Table 1 summarizes the feature schema, observed missingness across the 920 records, and enforced microservice physiological bounds. Substantial missingness occurs in fluoroscopy vessel count (66.41%), thalassemia defect status (52.83%), and exercise ST slope (33.59%).


**Table 1.** Clinical input schema, observed missingness (n = 920), and API validation bounds.

| Feature | Type | Missing (Count / %) | API Domain / Physiological Bound |
| --- | --- | --- | --- |
| age | Num. | 0 (0.0%) | Integer [18, 120] years |
| sex | Cat. | 0 (0.0%) | {Male, Female} |
| cp (chest pain) | Cat. | 0 (0.0%) | {typical, atypical, non-anginal, asympt.} |
| trestbps | Num. | 59 (6.4%) | Float [50.0, 250.0] mm Hg |
| chol | Num. | 30 (3.3%) | Float [50.0, 700.0] mg/dl |
| fbs (>120 mg/dl) | Bool. | 90 (9.8%) | Boolean {True, False} |
| restecg | Cat. | 2 (0.2%) | {normal, lv hypertrophy, st-t abnormality} |
| thalch (max HR) | Num. | 55 (6.0%) | Float [50.0, 250.0] bpm |
| exang | Bool. | 55 (6.0%) | Boolean {True, False} |
| oldpeak | Num. | 62 (6.7%) | Float [-5.0, 10.0] ST depression |
| slope | Cat. | 309 (33.6%) | {upsloping, flat, downsloping} |
| ca (vessels) | Num. | 611 (66.4%) | Float [0.0, 4.0] major vessels |
| thal | Cat. | 486 (52.8%) | {normal, fixed defect, reversable defect} |


### II. Leak-Free Preprocessing Pipeline

To guarantee zero statistical leakage from the evaluation cohort, the 920 records undergo a stratified 80/20 split (random seed 42) prior to imputation or scaling, yielding 736 training patients and 184 held-out test patients. Inside a composite scikit-learn ColumnTransformer, numerical attributes receive median imputation followed by z-score standardization, while categorical variables undergo most-frequent imputation followed by one-hot encoding configured to tolerate unseen categories during serving.


## Comparative Modeling and Hyperparameter Optimization


### I. Evaluation Criteria and Five-Algorithm Benchmarking

In diagnostic screening, false negatives correspond to discharging or delaying care for patients with underlying coronary disease. Model selection therefore balances overall F1-score, defined as the harmonic mean of precision P and recall R in (1), with high sensitivity and strong ranking discrimination measured by ROC-AUC:

$$
F_1 = \frac{2 \cdot P \cdot R}{P + R} \qquad (1)
$$

Five classification families—Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, Random Forest, and Support Vector Machine (SVM) with probability calibration—were wrapped inside end-to-end preprocessing pipelines and evaluated both under default configurations and after five-fold cross-validated GridSearchCV optimization maximizing F1-score.


### II. Optimization Results and Final Model Selection

Table 2 reports pre-tuning baseline performance, optimal GridSearchCV hyperparameters, and post-tuning test metrics across all five models on the 184 held-out patients. At baseline, SVM (F1 = 0.8664, recall = 0.9216) and Random Forest (accuracy = 0.8478, F1 = 0.8654, ROC-AUC = 0.9200) led the benchmark. After hyperparameter search, Tuned Random Forest (100 trees, unrestricted maximum depth, minimum samples split 5; cross-validation F1 = 0.8427) achieved the strongest overall test performance across accuracy (0.8696), precision (0.8611), F1-score (0.8857), and ROC-AUC (0.9180), alongside 0.9118 recall. Tuned SVM (C = 1, radial basis kernel) maintained the highest sensitivity at 0.9216 with 0.8664 F1-score. Figure 1 displays the five-model ROC comparison and the final Tuned Random Forest confusion matrix, confirming 93 true positives, 67 true negatives, 15 false positives, and only 9 missed disease cases out of 102 positive patients.


**Table 2.** Baseline and five-fold GridSearchCV tuned model performance on held-out test patients (n = 184).

| Model | Stage / Best GridSearchCV Params | Acc. | Prec. | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Random Forest | Baseline (Default) | 0.8478 | 0.8491 | 0.8824 | 0.8654 | **0.9200** |
| SVM | Baseline (Default) | 0.8424 | 0.8174 | **0.9216** | 0.8664 | 0.9167 |
| Logistic Reg. | Baseline (Default) | 0.8424 | 0.8411 | 0.8824 | 0.8612 | 0.9035 |
| KNN | Baseline (Default) | 0.8261 | 0.8302 | 0.8627 | 0.8462 | 0.8780 |
| Decision Tree | Baseline (Default) | 0.7554 | 0.7714 | 0.7941 | 0.7826 | 0.7507 |
| Random Forest | Tuned (trees=100, split=5, depth=None) | **0.8696** | **0.8611** | 0.9118 | **0.8857** | 0.9180 |
| SVM | Tuned (C=1, kernel=rbf) | 0.8424 | 0.8174 | **0.9216** | 0.8664 | 0.9167 |
| Logistic Reg. | Tuned (C=0.1) | 0.8315 | 0.8318 | 0.8725 | 0.8517 | 0.9023 |
| KNN | Tuned (k=11, weights=uniform) | 0.8261 | 0.8302 | 0.8627 | 0.8462 | 0.8959 |
| Decision Tree | Tuned (depth=5, split=10) | 0.7717 | 0.7941 | 0.7941 | 0.7941 | 0.8336 |


**Figure 1.** Diagnostic performance evaluation on the 184-patient test cohort. Panel (a) compares Receiver Operating Characteristic curves across all five classifiers; panel (b) shows the final Tuned Random Forest confusion matrix with 93 true positives and 9 false negatives.


## Clinical MLOps Architecture and Microservice Deployment

To operationalize the selected Random Forest architecture with full provenance and clinical safety guards, the workflow integrates experiment tracking, deterministic reproducibility checks, model registry gating, and Dockerized serving as depicted in Figure 2.


**Figure 2.** End-to-end clinical MLOps architecture for heart disease prediction, linking multi-center preprocessing, five-model cross-validated selection, MLflow governance, and Dockerized FastAPI risk stratification.


### I. MLflow Experiment Tracking, Reproducibility, and Registry Policy

Sixteen tracking runs recorded in the project SQLite-backed MLflow store under experiment Heart_Disease_Prediction verify both parameter sensitivity and deterministic execution. As shown in Table 3, RandomForest_Baseline (100 estimators, maximum depth 10, balanced class weight, seed 42) and repeated Reproducibility_Run_1 and Reproducibility_Run_2 trials yielded identical results across every metric (0.8641 accuracy, 0.8667 precision, 0.8922 recall, 0.8792 F1-score, and 0.9119 ROC-AUC). A deeper candidate configuration (RandomForest_Registry_V2 with 200 estimators and depth 15) scored lower at 0.8370 accuracy and 0.8544 F1-score across four registered versions of Heart_Disease_Production_Model; consequently, version 1 remains in Production while versions 2 through 4 are archived. Project configuration enforces a clinical recall quality gate of 0.75 before deployment packaging.


**Table 3.** MLflow experiment runs and registered clinical model versions.

| Tracked Artifact / Run | Accuracy | Precision | Recall | F1-Score | Status |
| --- | --- | --- | --- | --- | --- |
| Baseline / Repro Runs 1–2 (Seed 42) | **0.8641** | **0.8667** | **0.8922** | **0.8792** | Deterministic Match |
| Registry V2 (200 Trees, Depth 15) | 0.8370 | 0.8462 | 0.8627 | 0.8544 | Archived (v2–v4) |
| Deployed Production Bundle v1 | 0.8478 | 0.8491 | 0.8824 | 0.8654 | Production |


### II. Containerized FastAPI Service, Risk Stratification, and Verification

The Production bundle (serialized model, preprocessor, and lineage metadata) is served via a FastAPI application containerized on a python:3.11-slim base image exposing port 8001. Rather than returning raw binary integers alone, the inference endpoint translates predicted heart disease probability p into both a clinical diagnosis message and a three-tier clinical risk level Risk(p):

$$
\text{Risk}(p) = \begin{cases} \text{Low}, & p < 0.35 \\ \text{Moderate}, & 0.35 \le p < 0.65 \\ \text{High}, & p \ge 0.65 \end{cases} \qquad (2)
$$

Input validation enforces Pydantic boundary constraints across all 13 clinical variables—for example, age between 18 and 120 years, resting blood pressure between 50 and 250 millimeters of mercury, cholesterol between 50 and 700 milligrams per deciliter, and valid chest pain and thalassemia categories—returning structured HTTP 422 responses when violated and HTTP 503 guards if artifacts are unloaded. Both single (/predict) and cohort (/predict/batch) endpoints log structured audit telemetry. Execution of the automated test suite verified eight out of eight checks passing (root redirect, health status, metadata inspection, low-risk and high-risk patient predictions, invalid sex rejection, extreme blood pressure 999.0 rejection, and multi-patient batch scoring), complemented by a dedicated container-parity harness.


## Conclusion and Future Work

This study demonstrated an end-to-end clinical cardiovascular risk prediction platform uniting leak-free preprocessing of multi-center diagnostic cohorts, five-model hyperparameter optimization, deterministic MLflow tracking, and safety-validated FastAPI container deployment. Tuned Random Forest achieved 0.8696 accuracy, 0.9118 recall, and 0.8857 F1-score while limiting missed positive diagnoses to nine out of 102 test cases. Future extensions will incorporate probability calibration curves (such as Platt scaling or isotonic regression), SHAP-based per-patient clinical feature attributions in API responses, and prospective longitudinal drift auditing across hospital sites.


## References

[1] R. Detrano et al., “International application of a new probability algorithm for the diagnosis of coronary artery disease,” Amer. J. Cardiol., vol. 64, no. 5, pp. 304–310, Aug. 1989.

[2] D. Sculley et al., “Hidden technical debt in machine learning systems,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 28, pp. 2503–2511, 2015.

[3] M. Zaharia et al., “Accelerating the machine learning lifecycle with MLflow,” IEEE Data Eng. Bull., vol. 41, no. 4, pp. 39–45, Dec. 2018.

[4] A. Janosi et al., “Heart Disease,” UCI Machine Learning Repository, 1988. [Online]. Available: archive.ics.uci.edu/dataset/45/heart+disease.

[5] L. Breiman, “Random forests,” Mach. Learn., vol. 45, no. 1, pp. 5–32, Oct. 2001.

[6] C. Cortes and V. Vapnik, “Support-vector networks,” Mach. Learn., vol. 20, no. 3, pp. 273–297, Sep. 1995.

[7] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Mach. Learn. Res., vol. 12, pp. 2825–2830, Nov. 2011.

[8] S. Ramirez, FastAPI Framework Documentation. [Online]. Available: fastapi.tiangolo.com.


## About the Authors

**Author One** is a researcher focusing on clinical decision support systems, interpretable ensemble modeling, and reproducible healthcare machine learning pipelines.

**Author Two** specializes in medical data preprocessing, missing-value imputation strategies, and rigorous schema validation for diagnostic microservices.

**Author Three** researches containerized model serving, experiment provenance tracking, and automated testing for regulated machine learning operations.
