import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


celsius = np.array([-16, 31, -10, 37, -24, 28, 47, 44, -15, 34], dtype=float)
fahrenheit = np.array([3.2, 87.8, 14.0, 98.6, -11.2, 82.4, 116.6, 111.2, 5.0, 93.2], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss="mean_squared_error")

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius,fahrenheit, epochs=600, verbose=False)
print("Modelo entrenado ;)")

plt.xlabel("Epocas")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
plt.show()


entrada = np.array([[float(input("Ingrese el numero a convertir: "))]])
resultado = modelo.predict(entrada)
print("El resultado es " + str(resultado[0][0]) + " fahrenheit!")

pesos, sesgos = capa.get_weights()

print("Variables internas del modelo:")
print("Pesos de la conexión: \n", pesos)
print("Sesgo de la capa: \n", sesgos)