# Response Generation 

Scripts to train Seq2Seq and Transformer models on the Amazon Topical-Chat Corpus. This code serves as the baseline for [DSTC9 Track 3](http://dialog.speech.cs.cmu.edu:8003/).

**To train**: `python3 train.py --use_knowledge --transformer --save_path transformer/`

**To test**: `python3 test.py --use_knowledge --transformer --save_path transformer/`

**To serve interactive model with TF-IDF based fact selection**: `python3 dynamic.py --use_knowledge --transformer --save_path transformer/`

# Data

The pre-processed data can be found in `data.zip`. If you would like to use a different pre-processing strategy, please download the original data from [here](https://github.com/alexa/alexa-prize-topical-chat-dataset/).

# Pre-trained models

The pre-trained models can be found at: https://drive.google.com/file/d/1fPB45RDs_BcJ8KZeYQiauK3W1RsdY2hM/view?usp=sharing

```shell script
python3 test.py --save_path /docker-share/data/transformer_trained --epoch 20
test_freq Epoch: 20 PPL: 18.540181650272256
test_rare Epoch: 20 PPL: 26.536356859318733
```

# Contact

If you experience any issues with this code, please contact me at mehrishikib@gmail.com


# Results

* train.py
```shell script
python3 train.py --use_knowledge --transformer --batch_size=8 --save_path transformer/
...
Epoch 20/20 Batch 22450/22464 Avg Loss 3.81 LR 0.0002                                                                                                                                                                                         
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 22464/22464 [35:46<00:00, 10.47it/s]
```
* test.py
```shell script
CUDA_VISIBLE_DEVICES=1 python3 test.py --save_path transformer-2 --epoch 0                                                                                  
test_freq Epoch: 0 PPL: 13.543523391295622                  
test_rare Epoch: 0 PPL: 16.720335006129222 
CUDA_VISIBLE_DEVICES=1 python3 test.py --save_path transformer-2 --epoch 2
test_freq Epoch: 2 PPL: 6.41167198939869
test_rare Epoch: 2 PPL: 7.645547600879482
```