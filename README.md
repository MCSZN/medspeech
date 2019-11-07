## Medspeech

This repo is the host of our research on speech recognition for a medical audio dataset. It focuses on limited resources training classification model based on speech data. We use binary networks and *larq* for that purpose.
[Larq](<https://larq.dev>) is a small third party project that is not as optimized and stable as Tensorflow & Kera if training does not run on your machine replace larq layers with Keras layers.

To run the models you should map your data with a clean csv in the following format:


| file_name       | prompt | set        |
|:----------------|:------:|:-----------|
| "s101.wav"      | 1      | "train"    |
| "t302.wav"      | 4      | "test"     |
| "v302.wav"      | 2      | "validate" |




Your data should be in the following arrangement:

```bash
data
	├── recordings
	│   ├── test => 't101.wav', 't102.wav', ..., 't999.wav'
	│   ├── train =>'s101.wav', 's102.wav', ..., 's9999.wav'
	│   └── validate => 'v101.wav', 'v102.wav', ..., 'v999.wav'
	└── clean.csv
```

You can either run the ** main.py ** in the scripts directly. 
Or you can run the ** medspeech.py ** file. 
Cleaning the data before running the models is crucial.

We train and design our models with the main.py file. 
Load the data with *pipeline.py*. 
We pip install the requirement automatically with 
```bash 
bash setup.sh
``` 
