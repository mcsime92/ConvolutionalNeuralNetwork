from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# Initialize CNN
classifier = Sequential()

### 1. Convolution
# Conv 2D because we work with images. Videos would be 3D because time dimension
# 32 filters, size 3x3
# input_shape: shape of input image. 3 because 3 colored channels (RedGreenBlue), (2 if black & white)
# Activation relu to make sure we don't have negative values for pixels.
classifier.add(Convolution2D(32, (3, 3), input_shape = (64,64,3), activation = 'relu'))

### 2. Pooling
# Using 2x2 table to pool
classifier.add(MaxPooling2D(pool_size = (2, 2)))

### 3. Flattening
classifier.add(Flatten)

### 4. Connection
# sigmoid function for binary outcome
classifier.add(Dense(units = 128, activation= 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

### 5. Compile
# Stochastic gradient descent ADAM
# Binary outcome, hence binary loss function
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

