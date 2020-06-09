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

model.py：The main part of the model, it is realized by pytorch.

train.py：Build model and train the parameters in the model.

metrics.py：Define various evaluation functions(prec, ap, ndcg and rr).

dataset_build.py：Build train set and test set for model.

You can run the model on music dataset by 
'''
python3 --dataset music'
'''
