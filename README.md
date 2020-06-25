# Response Generation 

Scripts to train Seq2Seq and Transformer models on the Amazon Topical-Chat Corpus. This code serves as the baseline for [DSTC9 Track 3](http://dialog.speech.cs.cmu.edu:8003/).

**To train**: `python3 train.py --use_knowledge --transformer --save_path transformer/`

**To test**: `python3 test.py --use_knowledge --transformer --save_path transformer/`

**To serve interactive model with TF-IDF based fact selection**: `python3 dynamic.py --use_knowledge --transformer --save_path transformer/`

# Data

The pre-processed data can be found in `data.zip`. If you would like to use a different pre-processing strategy, please download the original data from [here](https://github.com/alexa/alexa-prize-topical-chat-dataset/).

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
Frequent set evaluation                                
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:30<00:00, 46.03it/s]
Epoch: 1 PPL: 73.4431151327329                                 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:30<00:00, 45.84it/s]
Epoch: 2 PPL: 69.88122154180827
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:30<00:00, 45.59it/s]
Epoch: 3 PPL: 70.74530481589977
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 45.30it/s]
Epoch: 4 PPL: 68.95724751094409
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 45.20it/s]
Epoch: 5 PPL: 64.75757782716687
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 45.10it/s]
Epoch: 6 PPL: 62.18060523987642
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 45.02it/s]
Epoch: 7 PPL: 59.162627740962364
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 45.02it/s]
Epoch: 8 PPL: 56.53898061598383
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 44.98it/s]
Epoch: 9 PPL: 53.643292373435926
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 44.79it/s]
Epoch: 10 PPL: 52.98827587148658
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 44.83it/s]
Epoch: 11 PPL: 52.649242886051425
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 44.80it/s]
Epoch: 12 PPL: 50.98023383215989
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 44.84it/s]
Epoch: 13 PPL: 49.91380856646045
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 44.78it/s]
Epoch: 14 PPL: 48.40016662256367
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 44.80it/s]
Epoch: 15 PPL: 48.647555507147196
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 44.78it/s]
Epoch: 16 PPL: 47.87302347074927
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 44.82it/s]
Epoch: 17 PPL: 47.21108803025195
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 44.81it/s]
Epoch: 18 PPL: 47.00970904844294
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 44.83it/s]
Epoch: 19 PPL: 47.0715745807156
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1410/1410 [00:31<00:00, 44.87it/s]
Epoch: 20 PPL: 46.152454725293204
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1401/1401 [10:12<00:00,  2.29it/s]
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