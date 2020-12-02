import pandas
import matplotlib.pyplot as plt
import math
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# Convierte un array de valores en un dataset con forma matricial
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


if __name__ == '__main__':
    # Se fija la semilla aleatoria
    numpy.random.seed(7)

    # Se carga el dataset y se fija el tipo de dato de la columna 'Passengers'
    dataset = pandas.read_csv('airline-passengers.csv', usecols=['Passengers'], engine='python')
    dataset = dataset.astype('float32')

    # Se imprimen los primeros registros del dataset para comprobar que está bien cargado
    print('Registros iniciales del dataset: ')
    print(dataset.head())

    # Se crea una gráfica para representar el dataset original
    plt.figure(figsize=(18, 9))
    plt.title('Evolución número de pasajeros (en millares)')
    plt.xlabel('Registro', fontsize=12)
    plt.plot(dataset)
    plt.show()

    # Se normaliza el dataset entre 0 y 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Se divide el dataset en un conjunto de pruebas y un conjunto de entrenamiento
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print('\nTamaño del conjunto de pruebas:', len(train))
    print('Tamaño del conjunto de entrenamiento:', len(test))

    # Se crea una gráfica para representar los conjuntos de prueba y entrenamiento
    plt.figure(figsize=(18, 9))
    plt.title('Conjuntos de prueba y entrenamiento')
    plt.xlabel('Registro', fontsize=12)
    plt.plot(train)
    plt.plot(test)
    plt.show()

    # Se remodelan los datos a X=t y Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # La red LSTM espera que los datos de entrada (X) se proporcionen en una estructura de array concreta:
    # [observación, tiempo, caracteristicas], y los datos actualmente están en la forma [observación, características]
    # Transformamos los datos en el formato esperado por LSTM [observación, tiempo, caracteristicas]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Creamos y entrenamos la red LSTM
    # La red tiene 1 entrada y 4 bloques LSTM (neuronas) y una capa de salida que hace una única predicción
    # Utiliza la función de activación sigmode por defecto para los bloques LSTM.
    # La red se ha entrenado 100 épocas (número de veces que el algoritmo entrena sobre el conjunto de prueba)
    # El tamaño batch es de 1 (número de muestras sobre las que se trabaja antes de actualizar el modelo)
    # Para actualizar los pesos se usa el algoritmo de ADAM (en lugar de 'gradient descent').
    # Como función de
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # Una vez que se ha entrenado el modelo, se puede medir el rendimiento con los dataset de prueba y de entrenamiento
    # Se generan predicciones
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Se invierten las predicciones para asegurar que se usan las mismas unidades que en los datos de ejemplo
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # Se calcula el RMSE (raíz del error cuadrático medio)
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('\nPuntuación sobre el conjunto de prueba: %.2f RMSE' % trainScore)
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Puntuación sobre el conjunto de test: %.2f RMSE' % testScore)

    # Se desplazan las predicciones  de entrenamiento en el eje de las x para alinearlas con los datos originales
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # Se desplazan las predicciones  de prueba en el eje de las x para alinearlas con los datos originales
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    # Se pintan los datos originales y las predicciones
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.title('Predicciones del modelo')
    plt.xlabel('Registro', fontsize=12)
    plt.show()
