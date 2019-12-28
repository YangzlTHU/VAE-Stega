This code belongs to "VAE-Stega: Linguistic Steganography Based on Variational Auto-Encoder".

## Requirements

- python 3
- tensorflow = 1.12
- zhusuan = 0.3.1
- gensim

## Prepare
1.

## Training

Train VAE_LSTM-LSTM model

```bash
python vae_lstm-lstm.py
```

Train VAE_BERT-LSTM model

```bash
python vae_bert-lstm.py
```

## Generating

Change `utils/parameters.py` or `utils/parameters_bert.py` to generate text or stega text.

## References

- [Generating Sentences from a Continuous Space](http://arxiv.org/abs/1511.06349)
- [vae_for_text](https://github.com/yiyang92/vae_for_text)
- [bert-use-demo](https://github.com/huwenxianglyy/bert-use-demo)
