from scripts import pipeline
from scripts import main
import pandas as pd

make_dl = pipeline.make_dl
print("Dataloader loaded")

model = main.model
print("Model loaded to memory")

path = input("Enter the path to csv\n")

dataset = pd.read_csv(path)
dataset.prompt = dataset.prompt.astype("category")

# train the model
train_dl = make_dl(dataset=dataset, set_="train", bs=32, signal=main.melspectrogram)
model.fit_generator(train_dl, epochs=10, verbose=2, workers=2)

# test it
test_dl = make_dl(dataset=dataset, set_="test", bs=32, signal=main.melspectrogram)
metric = model.evaluate_generator(test_dl, verbose=2, workers=2)
print(metric)
