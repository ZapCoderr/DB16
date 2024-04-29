# Database Group 16

Our own implementation of paper ["Sequence-based Target Coin Prediction for Cryptocurrency Pump-and-Dump"](https://arxiv.org/pdf/2204.12929.pdf).


## Download dataset

``` 
gdown https://drive.google.com/uc?id=1u2Ichky12k-ZTHDhqgFLM5WzlH26JnKa
``` 

``` 
gdown https://drive.google.com/uc?id=1slLs-OqMqzLHrmvzbf8xlyP2zzDpIk1R
``` 

## Install environment

``` 
pip install -r requirements.txt
```

## Run experiments


``` 
python dnn_torch.py 
```

``` 
python gru_torch.py 
```

``` 
python lstm_torch.py 
```

``` 
python snn_torch.py 
```