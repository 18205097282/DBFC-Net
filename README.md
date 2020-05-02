# DBFC-Net
**An overview of our DBFC-Net：**
![Alt text](https://github.com/18205097282/DBFC-Net/blob/master/ScreenShots/1.png)
**Results：**

- The MAP scores on bi-modality fine-grained cross-media retrieval for DBFCNet and the baselines. 
![Alt text](https://github.com/18205097282/DBFC-Net/blob/master/ScreenShots/2.png)
- The MAP scores of our cosine+ method compared with the cosine similarity. 
![Alt text](https://github.com/18205097282/DBFC-Net/blob/master/ScreenShots/4.png)

# Installation

**Requirement**

- pytorch, tested on [v1.0]
- CUDA, tested on v9.0
- Language: Python 3.6

## 1. Download dataset

Please visit this [dataset](http://59.108.48.34/tiki/FGCrossNet/).

## 2. Download trained model
The trained models of our DBFC-Net framework can be downloaded from  [Baidu Cloud](https://pan.baidu.com/s/14_XWs5tR53KKG2hUHKadFQ).

## 3. Prepare audio data
python audio.py


## 4. Training
sh train.sh


## 5. Testing
sh test.sh


# Citing
```
@inproceedings{he2019fine,
    Author = {Xiangteng He, Yuxin Peng, Liu Xie},
    Title = {A New Benchmark and Approach for Fine-grained Cross-media Retrieval},
    Booktitle = {Proc. of ACM International Conference on Multimedia (ACM MM)},
    Year = {2019}
} 
```

# Contributing
For any questions, feel free to open an issue or contact us. ([hexiangteng@pku.edu.cn]())
