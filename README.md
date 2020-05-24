# ACAM model
* Usage
 `python3 main.py`

* structure
```
.
├── code
│   ├── dataset_build.py
│   ├── main.py
│   ├── metrics.py
│   ├── model.py
│   └── train.py
├── data
│   ├── movie
│   │   ├── data
│   │   └── pretrained_embeddings
│   └── music
│       ├── data
│       └── pretrained_embeddings
└── README.md
```

model.py：The main part of the model
train.py：Build the model and train the parameters in the model
metrics.py：Define various evaluation functions
dataset_build.py：Read data

