# VAE-Stega

This code belongs to "VAE-Stega: Linguistic Steganography Based on Variational Auto-Encoder".

## Requirements

- python 3
- tensorflow = 1.12
- zhusuan = 0.3.1
- gensim

## Prepare for VAE_LSTM-LSTM Model

- Download our corpus file [movie](https://drive.google.com/file/d/1LP4ZIZsHDRf2ZgiMIu2EAIex_iC5WGFM/view?usp=sharing) 
or [tweet](https://drive.google.com/file/d/12YDuBm29TPkgB-zOpuBBRBBELdjb0uNb/view?usp=sharing) and put it in `./data/`
- Modify `./utils/parameters.py` to adjust the hyperparameters

## Prepare for VAE_BERT-LSTM model

- Download our corpus file [movie](https://drive.google.com/file/d/1LP4ZIZsHDRf2ZgiMIu2EAIex_iC5WGFM/view?usp=sharing) 
or [tweet](https://drive.google.com/file/d/12YDuBm29TPkgB-zOpuBBRBBELdjb0uNb/view?usp=sharing) and put it in `./data/`
- Download "BERT-Base, Uncased model" ([bert](https://github.com/google-research/bert)) and put it in `./pre/uncased_L-12_H-768_A-12`
- Modify `./utils/parameters_bert.py` to adjust the hyperparameters


## Training or Generating

VAE_LSTM-LSTM model

```bash
python vae_lstm-lstm.py
```

VAE_BERT-LSTM model

```bash
python vae_bert-lstm.py
```

## References

This code is based on yiyang92's [vae_for_text](https://github.com/yiyang92/vae_for_text), 
google-research's [bert](https://github.com/google-research/bert) 
and huwenxianglyy's [bert-use-demo](https://github.com/huwenxianglyy/bert-use-demo). 
Many thanks!
