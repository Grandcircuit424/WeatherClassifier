from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sci
import matplotlib.pyplot as plt

def alert(row):
    if row['heat_alert']:
        return 'heat'
    elif row['flood_alert']:
        return 'flood'
    else:
        return 'other'

def main():
    rainData = pd.read_csv(r'C:\Users\Grand\source\repos\WeatherData\synthetic_weather_alerts_selected.csv')

    rainData = rainData.drop(['fog_alert','storm_alert'], axis=1)

    cm = rainData.corr()

    sns.heatmap(cm, annot=True, cmap='coolwarm', center=0)

    rainData['alert_type'] = rainData.apply(alert, axis=1)

    y = rainData['alert_type']

    scaler = StandardScaler()

    features = [
        'temperature_C', 'humidity_percent', 'wind_speed_ms', 'pressure_hPa',
        'cloud_percent', 'dew_point_C', 'uv_index_real', 'precip_mm',
        'visibility_km_real', 'air_quality_index_scaled'
    ]

    colors = ['Red', 'Blue', 'Green', 'Yellow', 'Black', 'Gray', 'Teal', 'Pink', 'Orange', 'Purple']

    plt.figure(figsize=(24, 18))
    plotIndex = 1

    '''
    for i in range(2):
        for j in range(5):
            plt.subplot(2, 5, plotIndex)
            plt.hist(rainData[features[plotIndex-1]], rwidth=.7, color=colors[plotIndex-1])
            plt.title(features[plotIndex-1])
            plotIndex+=1

    plt.show()

    Amount = rainData['alert_type'].value_counts()

    plt.bar(Amount.index, Amount.values)
    plt.title("Alert Type")
    plt.show()

    '''
    X = rainData[features]

    #Spilting Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    #Making the SGDClassifier
    sgd = SGDClassifier(loss='log_loss', alpha= .0002, max_iter=1, random_state=42, warm_start=True)

    train_accuracy = []
    test_accuracy = []
    loss_table = []

    eproch = 40

    #iterations
    for i in range(eproch):
        X_train_s, y_train = shuffle(X_train_s, y_train, random_state=42+i)

        sgd.partial_fit(X_train_s, y_train, classes=np.unique(y))

        preds = sgd.predict_proba(X_train_s)
        loss = log_loss(y_train, preds)
        loss_table.append(loss)

        y_pred = sgd.predict(X_train_s)
        y_predTest = sgd.predict(X_test_s)

        accuracy = accuracy_score(y_train, y_pred)
        accuracyTest = accuracy_score(y_test, y_predTest)

        train_accuracy.append(accuracy)
        test_accuracy.append(accuracyTest)
    

    #Accuracy Rate of test and training Data (Performance)
    plt.plot(range(eproch), train_accuracy, c='r', label="Training Data")
    plt.plot(range(eproch), test_accuracy, c='g', label="Testing Data")
    plt.title("Accuracy Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right', ncol=1, fontsize='medium', frameon=True, title='Lines')
    plt.show()

    #Learning Curve
    plt.plot(range(eproch), loss_table, c='orange')
    plt.title("Learning Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    #Chaos Matrix of Training Data
    cm = confusion_matrix(y_train, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=["flood", "heat", "other"], yticklabels=["flood", "heat", "other"])
    plt.title("Training Data Confusion Matrix")
    plt.show()

    y_pred = sgd.predict(X_test_s)

    #Chaos Matrix of Testing Data
    cm = confusion_matrix(y_test, y_pred) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=["flood", "heat", "other"], yticklabels=["flood", "heat", "other"])
    plt.title("Testing Data Confusion Matrix")
    plt.show()



main()