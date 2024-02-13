from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train):
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo

def make_predictions(modelo, X_train, X_test):
    predicciones_train = modelo.predict(X_train)
    predicciones_test = modelo.predict(X_test)
    return predicciones_train, predicciones_test