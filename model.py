from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv("./churn.csv");

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

y_pred = gb_model.predict(X_test)
print(classification_report(y_test, y_pred))

probabilities = gb_model.predict_proba(X_test)