import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Example customer and PEP data (for demonstration, real use: load from your sources)
customers = pd.DataFrame({
    'customer_id': [1],
    'name': ['Alice Johnson'],
    'dob': ['1975-03-22'],
    'age': [48],
    'gender': ['F'],
    'nationality': ['UK'],
    'country_residence': ['UK'],
    'address': ['London'],
    'place_of_birth': ['Manchester'],
    'occupation': ['Minister'],
    'employer': ['Ministry of Finance'],
    'role': ['Minister of Finance'],
    'passport': ['A1234567'],
    'national_id': ['987654321'],
    'tax_id': ['TX12345'],
    'spouse': ['John Johnson'],
    'children': ['Emily Johnson'],
    'parent': ['Robert Johnson'],
    'sibling': ['Sarah Johnson'],
    'associate': ['Michael Minister'],
})

pep_list = pd.DataFrame({
    'pep_name': ['Alice Johnson', 'A Johnson'],
    'dob': ['1975-03-22', '1975-03-22'],
    'age': [48, 48],
    'gender': ['F', 'F'],
    'nationality': ['UK', 'UK'],
    'country_residence': ['UK', 'UK'],
    'address': ['London', 'London'],
    'place_of_birth': ['Manchester', 'Manchester'],
    'occupation': ['Minister', 'Minister'],
    'employer': ['Ministry of Finance', 'Ministry of Finance'],
    'role': ['Minister of Finance', 'Minister of Finance'],
    'passport': ['A1234567', 'A1234567'],
    'national_id': ['987654321', '987654321'],
    'tax_id': ['TX12345', 'TX12345'],
    'spouse': ['John Johnson', 'John Johnson'],
    'children': ['Emily Johnson', 'Emily Johnson'],
    'parent': ['Robert Johnson', 'Robert Johnson'],
    'sibling': ['Sarah Johnson', 'Sarah Johnson'],
    'associate': ['Michael Minister', 'Michael Minister'],
    'pep_category': ['Domestic', 'Domestic'],
    'term_start': ['2010-01-01', '2010-01-01'],
    'term_end': ['2015-12-31', '2015-12-31'],
    'active': [0, 0], # 0 = former, 1 = active
})

def build_pep_features(customer, pep_row):
    features = {}

    # Name features
    features['name_exact_match'] = int(customer['name'].lower() == pep_row['pep_name'].lower())
    features['name_fuzzy_score'] = fuzz.token_sort_ratio(customer['name'].lower(), pep_row['pep_name'].lower())
    features['name_variant_match'] = int(any(
        customer['name'].lower() == variant.lower() for variant in [pep_row['pep_name'], pep_row.get('alias', '')]
    ))
    features['alias_match'] = int(customer['name'].lower() == pep_row.get('alias', '').lower())

    # Biographical features
    features['dob_exact_match'] = int(customer['dob'] == pep_row['dob'])
    features['dob_partial_match'] = int(customer['dob'][:7] == pep_row['dob'][:7])  # year+month
    features['age_match'] = int(abs(customer['age'] - pep_row['age']) <= 2)
    features['gender_match'] = int(customer['gender'] == pep_row['gender'])

    # Demographic
    features['nationality_match'] = int(customer['nationality'] == pep_row['nationality'])
    features['country_residence_match'] = int(customer['country_residence'] == pep_row['country_residence'])
    features['address_match'] = int(customer['address'].lower() == pep_row['address'].lower())
    features['place_of_birth_match'] = int(customer['place_of_birth'].lower() == pep_row['place_of_birth'].lower())

    # Professional & Political
    features['occupation_match'] = int(customer['occupation'].lower() == pep_row['occupation'].lower())
    features['employer_match'] = int(customer['employer'].lower() == pep_row['employer'].lower())
    features['role_match'] = int(customer['role'].lower() == pep_row['role'].lower())
    cust_year = int(customer['dob'][:4])
    pep_start = int(pep_row['term_start'][:4])
    pep_end = int(pep_row['term_end'][:4])
    features['term_year_match'] = int((pep_start <= cust_year <= pep_end))
    features['pep_category_match'] = int(pep_row['pep_category'] == 'Domestic')  # Example encoding

    # Family & Associates
    features['spouse_match'] = int(customer['spouse'].lower() == pep_row['spouse'].lower())
    features['child_match'] = int(customer['children'].lower() == pep_row['children'].lower())
    features['parent_match'] = int(customer['parent'].lower() == pep_row['parent'].lower())
    features['sibling_match'] = int(customer['sibling'].lower() == pep_row['sibling'].lower())
    features['associate_match'] = int(customer['associate'].lower() == pep_row['associate'].lower())

    # Unique IDs
    features['passport_match'] = int(customer['passport'] == pep_row['passport'])
    features['national_id_match'] = int(customer['national_id'] == pep_row['national_id'])
    features['tax_id_match'] = int(customer['tax_id'] == pep_row['tax_id'])

    # Other features
    features['active_pep_flag'] = int(pep_row['active'])
    features['years_since_pep'] = 2025 - int(pep_row['term_end'][:4])

    return features

# For demonstration, build features for each customer-PEP pair
feature_rows = []
labels = []
for _, cust in customers.iterrows():
    for _, pep in pep_list.iterrows():
        row_features = build_pep_features(cust, pep)
        feature_rows.append(row_features)
        # Simulate label: 1 if names and passport match, else 0
        label = 1 if (row_features['name_exact_match'] and row_features['passport_match']) else 0
        labels.append(label)

features_df = pd.DataFrame(feature_rows)
y = np.array(labels)

# ML Model: Gradient Boosting
# For demonstration, train on this synthetic data (expand with real data for production)
if features_df.shape[0] > 1:  # Train/test split only if more than one sample
    X_train, X_test, y_train, y_test = train_test_split(features_df, y, test_size=0.4, random_state=42, stratify=y)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    results = X_test.copy()
    results['true_label'] = y_test
    results['predicted_label'] = y_pred
    results['confidence_score'] = y_pred_prob

    print(features_df)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nResults with confidence scores:\n", results)
else:
    print("Not enough data to train/test, features:")
    print(features_df)
