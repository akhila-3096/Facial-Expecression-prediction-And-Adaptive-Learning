
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Example dataset
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_classes=5,
    n_informative=7,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train Random Forest
best_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

best_model.fit(X_train, y_train)

# Predict
y_pred = best_model.predict(X_test)

# Display results
print("\nCLASSIFICATION REPORT\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("Accuracy:", accuracy_score(y_test, y_pred))
