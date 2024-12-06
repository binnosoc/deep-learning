# Step 1: Import Necessary Libraries

import tensorflow as tf 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense 
import warnings
import numpy as np 

warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

#Step 2: Define the Input Layer

input_layer = Input(shape=(20,))
print(input_layer)

#Step 3: Add Hidden Layers

hidden_layer1 = Dense(64, activation='relu')(input_layer) 
hidden_layer2 = Dense(64, activation='relu')(hidden_layer1) 

#Step 4: Define the Output Layer

output_layer = Dense(1, activation='sigmoid')(hidden_layer2) 

#Step 5: Create the Model

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

#Step 6: Compile the Model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Step 7: Train the Model

X_train = np.random.rand(1000, 20) 
y_train = np.random.randint(2, size=(1000, 1)) 
model.fit(X_train, y_train, epochs=10, batch_size=32) 

#Step 8: Evaluate the Model

X_test = np.random.rand(200, 20) 
y_test = np.random.randint(2, size=(200, 1)) 
loss, accuracy = model.evaluate(X_test, y_test) 
print(f'Test loss: {loss}') 
print(f'Test accuracy: {accuracy}') 