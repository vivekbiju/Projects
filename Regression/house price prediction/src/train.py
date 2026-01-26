import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split

# 1. LOAD DATA
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_id = test['Id']

# 2. OUTLIER REMOVAL
train = train[train['GrLivArea'] < 4500]

# 3. TARGET TRANSFORMATION (Log-normalize SalePrice)
y = np.log1p(train["SalePrice"]).reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test.copy()

# 4. CONCATENATE FOR UNIFIED CLEANING
features = pd.concat((train_features, test_features)).reset_index(drop=True)

# 5. DATA CLEANING & IMPUTATION
def filling_null(features):
    # Manual filling based on documentation
    features['Functional'] = features['Functional'].fillna('Typ')
    
    # Mode Imputation for categorical
    mode_cols = ['Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
    for col in mode_cols:
        features[col] = features[col].fillna(features[col].mode()[0])
    
    # Contextual Imputation (Grouped)
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    
    # Bulk filling: 'None' for categorical, 0 for numerical
    cat_features = features.select_dtypes(include='object').columns
    num_features = features.select_dtypes(exclude='object').columns
    features[cat_features] = features[cat_features].fillna('None')
    features[num_features] = features[num_features].fillna(0)
    
    return features

features = filling_null(features)

# 6. SKEWNESS FIX (Box-Cox Transformation)
def fix_skew(features):
    numerical_columns = features.select_dtypes(exclude='object').columns
    skewed_features = features[numerical_columns].apply(lambda x: x.skew()).sort_values(ascending=False)
    
    high_skew = skewed_features[abs(skewed_features) > 0.5]
    for column in high_skew.index:
        features[column] = boxcox1p(features[column], boxcox_normmax(features[column] + 1))
    return features

features = fix_skew(features)

# 7. ENCODING CATEGORICAL VARIABLES
features = pd.get_dummies(features).reset_index(drop=True)

# 8. RE-SPLIT DATA INTO TRAIN AND TEST
X = features.iloc[:len(y), :]
X_test_final = features.iloc[len(y):, :]

# Split training data for internal validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. MODELING (Lasso)
model_lasso = Lasso(alpha=0.0005) # Optimized alpha
model_lasso.fit(X_train, y_train)

# 10. PREDICTION & SUBMISSION
# Predict on the competition test set
final_preds = model_lasso.predict(X_test_final)

# Convert back from Log scale to Dollars
final_preds = np.expm1(final_preds)

# Create submission file
submission = pd.DataFrame({'Id': test_id, 'SalePrice': final_preds})
submission.to_csv('submission.csv', index=False)

print("Project Complete! submission.csv has been created.") 
 
