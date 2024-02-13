import data_loader
import model
import evaluation
import visualization

def main():
    # Paso 1: Cargar los datos
    datos = data_loader.load_data('GBM.MX', start_date='2020-01-01', end_date='2023-01-01')

    # Paso 2: Preparar los datos
    X_train, X_test, y_train, y_test = data_loader.prepare_data(datos)

    # Paso 3: Entrenar el modelo
    modelo = model.train_model(X_train, y_train)

    # Paso 4: Hacer predicciones
    predicciones_train, predicciones_test = model.make_predictions(modelo, X_train, X_test)

    # Paso 5: Evaluar el modelo
    evaluation.evaluate_model(y_train, predicciones_train, y_test, predicciones_test)

    # Paso 6: Visualizar los resultados
    visualization.plot_results(datos, X_test, y_test, predicciones_test)

if __name__ == "__main__":
    main()