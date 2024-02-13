import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split

def load_data(ticker, start_date, end_date):
    datos = yf.download(ticker, start=start_date, end=end_date)
    datos.reset_index(inplace=True)
    datos['Fecha_Numérica'] = pd.to_numeric(datos['Date'])
    return datos

def prepare_data(datos):
    X = datos[['Fecha_Numérica']]
    y = datos['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test