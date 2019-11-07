import tensorflow as tf
from . import pipeline
import pandas as pd
import larq as lq
from librosa.feature import melspectrogram, mfcc
from numpy.fft import fft
from scipy.fftpack import dct
from time import time


# create our model / compile it
model = tf.keras.models.Sequential()
model.add(
    lq.layers.QuantConv2D(
        filters=15,
        kernel_size=5,
        padding="same",
        input_shape=(128, 459, 1,),
        activation="relu",
    )
)
model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 1)))
model.add(lq.layers.QuantConv2D(3, (4, 4), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))
model.add(lq.layers.QuantConv2D(2, (2, 2), activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Reshape((1, 6780,)))
model.add(tf.keras.layers.GRU(1, activation="tanh", return_sequences=True))
model.add(tf.keras.layers.Flatten())
model.add(lq.layers.QuantDense(25, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# print summary
print(model.summary())

if __name__ == "__main__":
	print('Executing training from script/main.py')
	# read csv
	dataset = pd.read_csv("../data/clean.csv")
	dataset.prompt = dataset.prompt.astype("category")
	# instantiate DataLoader
	train_loader = pipeline.make_dl(dataset=dataset, set_="train", bs=32, signal=melspectrogram, script=True)

	# fit the model on train loader data
	model.fit_generator(train_loader, epochs=10, verbose=2, workers=2)

	# instantiate test loader
	test_loader = pipeline.make_dl(dataset=dataset, set_="test", bs=32, signal=melspectrogram, script=True)
	# evaluate model on test data with the loader
	metric = model.evaluate_generator(test_loader, workers=2, verbose=2)
