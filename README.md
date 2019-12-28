This code belongs to "VAE-Stega: Linguistic Steganography Based on Variational Auto-Encoder".

## Requirements

- python 3
- tensorflow = 1.12
- zhusuan = 0.3.1
- gensim

## Prepare for VAE_LSTM-LSTM Model

- Put your corpus file in `data/train_yourdatasetname.txt`
- Modify `utils/parameters.py`

## Prepare for VAE_LSTM-LSTM model

- Put your corpus file in `data/train_yourdatasetname.txt`
- Download "BERT-Base, Uncased model" and put in `pre/uncased_L-12_H-768_A-12`
- Modify `utils/parameters_bert.py`


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

- [Generating Sentences from a Continuous Space](http://arxiv.org/abs/1511.06349)
- [vae_for_text](https://github.com/yiyang92/vae_for_text)
- [bert](https://github.com/google-research/bert)
- [bert-use-demo](https://github.com/huwenxianglyy/bert-use-demo)
