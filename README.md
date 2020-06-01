# VAE-Stega

This code belongs to "VAE-Stega: Linguistic Steganography Based on Variational Auto-Encoder".

## Requirements

- python 3
- tensorflow = 1.12
- zhusuan = 0.3.1
- gensim

## Prepare for VAE_LSTM-LSTM Model

- Download our corpus file [movie]() 
or [tweet]() and put it in `./data/`
- Modify `./utils/parameters.py` to adjust the hyperparameters

## Prepare for VAE_BERT-LSTM model

- Download our corpus file [movie](https://drive.google.com/open?id=1rsd0US0Xb4HUOOqjtZSr-M1XITU7jy4k) 
or [tweet](https://drive.google.com/open?id=1MPhE6RkIMUWIr2boN7Oc0X9knL-9Zs-t) and put it in `./data/`
- Download "BERT-Base, Uncased model" ([bert](https://github.com/google-research/bert)) and put it in `./pre/uncased_L-12_H-768_A-12`
- Modify `./utils/parameters_bert.py` to adjust the hyperparameters


## Training or Generating

- VAE_LSTM-LSTM model

```bash
python vae_lstm-lstm.py
```

- VAE_BERT-LSTM model

```bash
python vae_bert-lstm.py
```

## References

This code is based on yiyang92's [vae_for_text](https://github.com/yiyang92/vae_for_text), 
google-research's [bert](https://github.com/google-research/bert) 
and huwenxianglyy's [bert-use-demo](https://github.com/huwenxianglyy/bert-use-demo). 
Many thanks!
