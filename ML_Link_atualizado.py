import requests
import hashlib
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm


def download_file(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)


def get_file_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def convert_to_numeric(value):
    try:
        return float(value)
    except ValueError:
        return np.nan


def monitor_file_changes(url, file_path, interval=60):
    current_hash = None

    while True:
        download_file(url, file_path)
        new_hash = get_file_hash(file_path)

        if new_hash != current_hash:
            print("File has been updated. Performing analysis...")
            try:
                perform_statistical_analysis(file_path)
            except pd.errors.ParserError as e:
                print("Error parsing CSV file:", e)
                print("The file may be corrupted. Please check the source.")
            current_hash = new_hash

        time.sleep(interval)


def perform_statistical_analysis(file_path):
    dados_csv = pd.read_csv(file_path, converters={'personnel': convert_to_numeric})
    dados_csv.dropna(subset=['personnel'], inplace=True)

    X = dados_csv[['day']]
    X = sm.add_constant(X)
    y = dados_csv['personnel']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = sm.OLS(y_train, X_train).fit()

    print(modelo.summary())

    y_pred = modelo.predict(X_test)

    coeficiente_regressao_linear = modelo.params['day'] * 900000
    print("Índice multiplicativo da regressão linear (cor azul):", coeficiente_regressao_linear)

    plt.scatter(X_test['day'], y_test, color='black')
    plt.plot(X_test['day'], y_pred, color='blue', linewidth=3, label='Regressão Linear')

    resultado_filepath = 'resultado_regressao_linear.png'
    plt.savefig(resultado_filepath)

    print(f"Resultado da Regressão Linear salvo em: {resultado_filepath}")
    plt.show()


if __name__ == "__main__":
    url = "https://www.kaggle.com/datasets/piterfm/2022-ukraine-russian-war?select=russia_losses_personnel.csv"
    file_path = "russia_losses_personnel.csv"
    monitor_file_changes(url, file_path)




