import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size

    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)

        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)
    

def standardize(uni_data, TRAIN_SPLIT):

    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std  = uni_data[:TRAIN_SPLIT].std()

    uni_data = (uni_data - uni_train_mean) / uni_train_std
    
    return uni_data

def baseline(history):
  return np.mean(history)


def main():

    df = pd.read_csv('data/rune_data.csv')
    df.set_index('timestamp')

    tf.random.set_seed(13)
    
    univariate_past_history    = 20
    univariate_future_target   = 0

    BATCH_SIZE  = 20
    BUFFER_SIZE = 5
    
    uni_data       = df['Chaos_rune']
    uni_data.index = df['timestamp']

    uni_data = uni_data.values
    
    TRAIN_SPLIT = int(0.8 * len(uni_data)) # percent of dataset used for training
    
    uni_data = standardize(uni_data, TRAIN_SPLIT)

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT, univariate_past_history, univariate_future_target)
    x_val_uni, y_val_uni     = univariate_data(uni_data, TRAIN_SPLIT, None, univariate_past_history, univariate_future_target)
    
    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.Dense(1)
        ])

    simple_lstm_model.compile(optimizer='adam', loss='mae')

    
                                                        
if(__name__ == "__main__"):
    main()
