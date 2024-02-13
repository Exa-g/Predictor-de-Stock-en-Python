import matplotlib.pyplot as plt

def plot_results(datos, X_test, y_test, predicciones_test):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='black', label='Datos de prueba')
    plt.plot(X_test, predicciones_test, color='blue', linewidth=3, label='Predicciones')
    plt.title('Predicciones del modelo de regresión lineal')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de cierre')
    plt.legend()
    plt.show()