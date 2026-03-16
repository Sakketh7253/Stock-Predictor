import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

class BaselineLSTM:
    def __init__(self, input_shape, num_classes=3):
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.num_classes = num_classes
        self.name = "Baseline LSTM"
        self.input_shape = input_shape
        
    def fit(self, X, y, epochs=10, batch_size=32):
        # We expect X in shape (samples, features). We need (samples, 1, features)
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=1)
            
        # y comes as -1, 0, 1. Map to 0, 1, 2. Clip for safety.
        y_mapped = np.clip(np.array(y, dtype=np.int32) + 1, 0, self.num_classes - 1)
        y_cat = to_categorical(y_mapped, num_classes=self.num_classes)
        self.model.fit(X, y_cat, epochs=epochs, batch_size=batch_size, verbose=0)
        
    def predict(self, X):
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=1)
        y_pred_prob = self.model.predict(X, verbose=0)
        y_pred_mapped = np.argmax(y_pred_prob, axis=1).astype(np.int32)
        return y_pred_mapped - 1  # Unmap back to -1, 0, 1
