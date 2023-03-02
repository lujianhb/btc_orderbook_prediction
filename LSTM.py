import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
import os
from keras.engine import training_utils
from sklearn.utils import class_weight
from sklearn import preprocessing

SEQ_LEN = 16  # how long of a preceeding sequence to collect
FUTURE_PERIOD_PREDICT = 12  # how far into the future are we trying to predict
VALIDATION_SPLIT = 0.1

EPOCHS = 10  # how many passes through our data
BATCH_SIZE = 1  # 128  # 64

NNNAME = "LSTM_STATEFUL"

NAME = f"{NNNAME}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

SAVE = True

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# %%
# classify based on:
#   * midpoint of best ask & best bid  later vs now
#   * buy at best ask & sell at best bid
#   * buy at last buy price & sell at last sell price
#   * can make profitable trade from OB other side

# changable:
#   * threshold
#   * how far into the future to compare

def classify(df):
    change = 0.0015
    for i, row in df.iterrows():
        target = 1
        if df.at[i, "future_last_sell_price"] > df.at[i, "last_buy_price"] * (1 + change):  # good to buy now
            target = 2
        elif df.at[i, "future_last_buy_price"] * (1 + change) < df.at[i, "last_sell_price"]:  # good to sell now
            target = 0
        df.at[i, "target"] = target


# %%
# Preprocessing the dataset to a uniform output format
def preprocess(df, shorten=False):
    df = df.loc[:, ~df.columns.str.startswith('future_')]

    sprd = 2 * (df["best_ask"] - df["best_bid"]) / (df["best_ask"] + df["best_bid"])
    df.insert(2, "spread", sprd)

    ob_maxsum = df.filter(regex="(a|b)\d").sum(axis=1).max()
    ob_maxamt = df.filter(regex="last_buy_amt|last_sell_amt").sum(axis=1).max()

    for col in df.columns:
        if col in ["target", "spread"]:  # normalize all except for target, spread already done
            continue

        if col in ["best_ask", "best_bid", "last_buy_price", "last_sell_price"]:
            df[col] = df[col].pct_change()
        elif col in ["last_buy_amt", "last_sell_amt"]:
            df[col] = df[col] / ob_maxamt  # preprocessing.scale(df[col].values)
        else:
            df[col] = df[col] / ob_maxsum

        df.dropna(inplace=True)  # remove the nas created by pct_change

    df.dropna(inplace=True)  # cleanup again... jic.

    prev_days = deque(maxlen=SEQ_LEN)

    X = []
    y = []

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            X.append(np.array(prev_days))  # x is the sequences
            y.append(i[-1])  # y is the targets/labels

    return np.array(X), y


# %%
path = "/Users/Ranykhalil/Downloads/Archive_2/"


def read_ob_csv(file):
    df = pd.read_csv(path + file)
    df.set_index("time", inplace=True)
    df_len = len(df.index.unique())
    df_interval = df.index.unique()[1] - df.index.unique()[0]
    df_init_t = df.index.unique()[0]
    df_bins = len(df.loc[df.index == df_init_t].columns) - 6
    return df_len, df_interval, df_init_t, df_bins, df


data_file = "obbinsandtrades_BTC-USD_2020-03-26-23.49.57.83_2020-04-03-15.49.47.83_17196840.csv"
df_len, df_interval, df_init_t, df_bins, df = read_ob_csv(data_file)

df.fillna(method="ffill", inplace=True)
df.dropna(inplace=True)

df.head()
# %%
if "future_best_ask" not in df.columns:
    for col in df.columns:
        df['future_' + col] = df[col].shift(-FUTURE_PERIOD_PREDICT)

classify(df)
df.dropna(inplace=True)
df.groupby("target").count()
# %%
times = sorted(df.index.values)
validation_times = sorted(df.index.values)[-int(VALIDATION_SPLIT * len(times))]

validation_df = df[(df.index >= validation_times)]
df = df[(df.index < validation_times)]

train_x, train_y = preprocess(df, False)
validation_x, validation_y = preprocess(validation_df, False)

print("shape tr x", np.shape(train_x))
print("shape tr y", np.shape(train_y))
print("shape val x", np.shape(validation_x))
print("shape val y", np.shape(validation_y))
print(f"""###
### TRAIN      \t total: {len(train_x)} \t holds: {train_y.count(1)} \t buys: {train_y.count(2)} \t sells: {train_y.count(0)}
### VALIDATION \t total: {len(validation_x)} \t holds: {validation_y.count(1)} \t buys: {validation_y.count(2)} \t sells: {validation_y.count(0)}
###""")

model = Sequential()
model.add(LSTM(128, batch_input_shape=(BATCH_SIZE, train_x.shape[1], train_x.shape[2]), input_shape=(train_x.shape[1:]),
               return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, batch_input_shape=(BATCH_SIZE, train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, batch_input_shape=(BATCH_SIZE, train_x.shape[1], train_x.shape[2]), ))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['sparse_categorical_accuracy']
)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# unique file name that will include the epoch and the validation acc for that epoch
filepath = NNNAME + "-{epoch:02d}-{val_acc:.3f}"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_sparse_categorical_accuracy',
                                                      verbose=1, save_best_only=True,
                                                      mode='max'))  # saves only the best ones
train_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(train_y),
                                                  train_y)
print("train_weights", train_weights)
# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=False,
    validation_data=(validation_x, validation_y),
    # class_weight=dict(enumerate(train_weights)),
    callbacks=[tensorboard, checkpoint],
    class_weight=train_weights,
)
val_weights = class_weight.compute_class_weight('balanced',
                                                np.unique(validation_y),
                                                validation_y)

val_sample_weights = training_utils.standardize_weights(np.array(validation_y),
                                                        class_weight=dict(enumerate(val_weights)))

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0, sample_weight=val_sample_weights)
# Save model
if SAVE:
    model.save("models/{}".format(NAME))
    print(f'### MODEL: \t {NAME}')
else:
    print('[model not saved]')
print('### Test loss: \t', score[0])
print('### Test accuracy: \t', score[1])
# %% md
