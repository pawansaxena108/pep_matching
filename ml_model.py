import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Dummy data setup
data = {
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['John Smith', 'Alice Johnson', 'Robert Downey', 'Maria Garcia', 'Michael Minister'],
    'nationality': ['USA', 'UK', 'USA', 'Spain', 'UK'],
    'occupation': ['Engineer', 'Politician', 'Actor', 'Teacher', 'Minister']
}

customers = pd.DataFrame(data)

# Dummy PEP names list simulating parsed XML
pep_names = ['alice johnson', 'michael minister']

# Define high risk occupations
high_risk_occupations = {'politician', 'minister', 'ambassador', 'government official'}

# Rule-based flagging function
def rule_based_flags(row, pep_names, high_risk_occupations):
    name = row['name'].lower()
    name_match_scores = [fuzz.token_sort_ratio(name, pep_name) for pep_name in pep_names]
    max_score = max(name_match_scores) if name_match_scores else 0
    name_flag = 1 if max_score >= 85 else 0
    occupation_flag = 1 if row['occupation'].lower() in high_risk_occupations else 0
    rule_score = name_flag + occupation_flag
    return pd.Series({
        'rule_score': rule_score,
        'name_match_score': max_score,
        'occupation_flag': occupation_flag
    })

# Apply rule-based logic
customers[['rule_score', 'name_match_score', 'occupation_flag']] = customers.apply(
    rule_based_flags, axis=1, pep_names=pep_names, high_risk_occupations=high_risk_occupations)

# Create synthetic labels (in reality, use real labels)
customers['is_pep'] = np.where(customers['rule_score'] > 0, 1, 0)

# Encoding categorical features for ML
le_nat = LabelEncoder()
le_occ = LabelEncoder()
customers['nat_encoded'] = le_nat.fit_transform(customers['nationality'])
customers['occ_encoded'] = le_occ.fit_transform(customers['occupation'])

# Features and label
features = ['name_match_score', 'occupation_flag', 'nat_encoded', 'occ_encoded']
X = customers[features]
y = customers['is_pep']

# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)


# Train ML model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and score
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Results
results = X_test.copy()
results['true_label'] = y_test.values
results['predicted_label'] = y_pred
results['confidence_score'] = y_pred_prob

print(customers)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nResults with confidence scores:\n", results)
