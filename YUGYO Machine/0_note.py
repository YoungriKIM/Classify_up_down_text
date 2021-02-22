from tensorflow.keras.layers import Embedding, Dense, LSTM, GlobalAveragePooling1D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

vocab_size = 572 

# model = Sequential()
# model.add(Embedding(vocab_size, input_length=18, output_dim=100))
# model.add(LSTM(128))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()


model = Sequential()
model.add(Embedding(vocab_size, input_length=18, output_dim=100))
model.add(LSTM(128, activation='relu'))
model.add(Dense(64))
model.add(Dense(1, activation='sigmoid'))
model.summary()
