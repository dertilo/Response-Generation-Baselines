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
(base) root@gunther:/docker-share/Response-Generation-Baselines# CUDA_VISIBLE_DEVICES=1 python3 test.py --use_knowledge --transformer --batch_size=8 --save_path /docker-share/data/transformer_trained
Number of training instances: 179709
Number of validation (freq) instances: 11274
Number of validation (rare) instances: 11246
Number of testing (freq) instances: 11207
Number of testing (rare) instances: 11206
100%|███████| 1410/1410 [07:02<00:00,  3.34it/s]
Epoch: 20 PPL: 20.090918719325273

(base) root@gunther:/docker-share/Response-Generation-Baselines# python run_metrics.py --save_path /docker-share/data/transformer_trained/
F-1 score: 0.1486375601119171
Distinct Unigrams: 0.7794670809070053
Distinct Bigrams: 0.8808489533312245
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
(base) root@gunther:/docker-share/Response-Generation-Baselines# python3 test.py --use_knowledge --transformer --batch_size=8 --save_path transformer/
Number of training instances: 179709                
Number of validation (freq) instances: 11274           
Number of validation (rare) instances: 11246              
Number of testing (freq) instances: 11207             
Number of testing (rare) instances: 11206               
Frequent set evaluation for 41 epochs
100%|████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 45.48it/s]
Epoch: 1 PPL: 69.01204057678481
100%|████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 45.40it/s]
Epoch: 2 PPL: 67.77956874669752
100%|████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 45.32it/s]
Epoch: 3 PPL: 63.137275843614844
100%|████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 45.04it/s]
...
Epoch: 39 PPL: 38.044889044852134
100%|████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 45.14it/s]
Epoch: 40 PPL: 38.48767700783606
100%|████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 45.08it/s]
Epoch: 41 PPL: 37.944097115863094
```
* run_metrics.py
```shell script
(base) root@gunther:/docker-share/Response-Generation-Baselines# python run_metrics.py --save_path transformer/
F-1 score: 0.17567660378679692
Distinct Unigrams: 0.6508990395535111
Distinct Bigrams: 0.7625310789845575
Traceback (most recent call last):
  File "run_metrics.py", line 49, in <module>
    rare_out = [l.strip().split() for l in open(args.save_path + 'rare_out.tgt').readlines()]
FileNotFoundError: [Errno 2] No such file or directory: 'transformer/rare_out.tgt'
```