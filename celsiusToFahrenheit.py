import tensorflow as tf
import numpy as np

## TRAIN DATA

# Celsius and Fahrenheit Array 
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

## STRUCTURE 

# Layer and Model
output_layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential(output_layer)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

## TRAINING DATA
print("Comenzando el entrenamiento...")
history = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado!")

## LOSS FUNCTION
import matplotlib.pyplot as plt
plt.xlabel("# Epochs")
plt.ylabel("Loss")
plt.plot(history.history["loss"])

##PREDICTIONS
print("Hagamos una predicción!")
input_data = 100.0
result = model.predict([input_data])
print("El resultado es " + str(input_data) + " celsius corresponden a " + str(result) + " fahrenheit!")

##INTERNAL VALUES
print("Variables internas del modelo")
print(output_layer.get_weights())

