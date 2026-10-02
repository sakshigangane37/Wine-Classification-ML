import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ------------------------------------------------------------
# Display Border
# ------------------------------------------------------------

def Border():
    print("-" * 70)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

def LoadData(filename):
    Data = pd.read_csv(filename)
    return Data

# ------------------------------------------------------------
# Clean and Prepare Data
# ------------------------------------------------------------

def PrepareData(Data):
    print("Missing Values :", Data.isnull().sum().sum())
    print("Duplicate Rows :", Data.duplicated().sum())

    Features = [
        "Alcohol",
        "Malic acid",
        "Ash",
        "Alcalinity of ash",
        "Magnesium",
        "Total phenols",
        "Flavanoids",
        "Nonflavanoid phenols",
        "Proanthocyanins",
        "Color intensity",
        "Hue",
        "OD280/OD315 of diluted wines",
        "Proline"
    ]

    X = Data[Features]
    Y = Data["Class"]

    return X, Y

# ------------------------------------------------------------
# Split Dataset
# ------------------------------------------------------------

def SplitData(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    return X_train, X_test, Y_train, Y_test

# ------------------------------------------------------------
# Train Model
# ------------------------------------------------------------

def TrainModel(X_train, Y_train):
    Model = DecisionTreeClassifier(random_state=42)
    Model.fit(X_train, Y_train)
    return Model

# ------------------------------------------------------------
# Test Model
# ------------------------------------------------------------

def TestModel(Model, X_test):
    Y_pred = Model.predict(X_test)
    return Y_pred

# ------------------------------------------------------------
# Calculate Accuracy
# ------------------------------------------------------------

def CalculateAccuracy(Y_test, Y_pred):
    Accuracy = accuracy_score(Y_test, Y_pred)

    Border()
    print("Model Accuracy")
    Border()
    print("Testing Accuracy :", Accuracy)
    print("Testing Accuracy Percentage :", Accuracy * 100, "%")

    return Accuracy

# ------------------------------------------------------------
# Display Confusion Matrix
# ------------------------------------------------------------

def DisplayConfusionMatrix(Y_test, Y_pred):
    Border()
    print("Confusion Matrix")
    Border()

    Matrix = confusion_matrix(Y_test, Y_pred)
    print(Matrix)

# ------------------------------------------------------------
# Display Classification Report
# ------------------------------------------------------------

def DisplayClassificationReport(Y_test, Y_pred):
    Border()
    print("Classification Report")
    Border()
    print(classification_report(Y_test, Y_pred))

# ------------------------------------------------------------
# Display Predictions
# ------------------------------------------------------------

def DisplayPredictions(Y_test, Y_pred):
    Border()
    print("Actual and Predicted Results")
    Border()

    Result = pd.DataFrame({
        "Actual Class": Y_test.values,
        "Predicted Class": Y_pred
    })

    print(Result.to_string(index=False))

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------

def main():
    Border()
    print("Wine Classification Machine Learning")
    Border()

    Data = LoadData("WinePredictor.csv")

    print("Dataset Loaded Successfully")
    print("Dataset Shape :", Data.shape)
    print("Dataset Columns :", len(Data.columns))

    Border()
    print("Data Preparation")
    Border()

    X, Y = PrepareData(Data)

    print("Number of Features :", X.shape[1])
    print("Number of Classes :", Y.nunique())
    print("Class Distribution :")
    print(Y.value_counts().sort_index())

    X_train, X_test, Y_train, Y_test = SplitData(X, Y)

    Border()
    print("Training Data")
    Border()
    print("Training Samples :", len(X_train))

    Border()
    print("Testing Data")
    Border()
    print("Testing Samples :", len(X_test))

    Model = TrainModel(X_train, Y_train)

    Border()
    print("Decision Tree Model Trained Successfully")
    Border()

    Y_pred = TestModel(Model, X_test)

    DisplayPredictions(Y_test, Y_pred)
    CalculateAccuracy(Y_test, Y_pred)
    DisplayConfusionMatrix(Y_test, Y_pred)
    DisplayClassificationReport(Y_test, Y_pred)

    Border()
    print("Wine Classification Completed Successfully")
    Border()

if __name__ == "__main__":
    main()