from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(y_train, predicciones_train, y_test, predicciones_test):
    mse_train = mean_squared_error(y_train, predicciones_train)
    r2_train = r2_score(y_train, predicciones_train)
    mse_test = mean_squared_error(y_test, predicciones_test)
    r2_test = r2_score(y_test, predicciones_test)
    print("Entrenamiento - Error Cuadrático Medio:", mse_train)
    print("Entrenamiento - Coeficiente de Determinación (R^2):", r2_train)
    print("Prueba - Error Cuadrático Medio:", mse_test)
    print("Prueba - Coeficiente de Determinación (R^2):", r2_test)