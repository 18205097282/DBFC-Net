Introduction
---
Datasets and source code for our paper **DBFC-Net: Salvage DBFC-Net: A Uniform Framework for Fine-grained Cross-media Retrieval**


Network Architecture
---
![Alt text](https://github.com/18205097282/DBFC-Net/blob/master/ScreenShots/1.png)


Installation
---

 ```
- pytorch, tested on [v1.0]
- CUDA, tested on v9.0
- Language: Python 3.6
 ```
- Data Preparation
Please visit this [dataset](http://59.108.48.34/tiki/FGCrossNet/).

- Demo model
The trained models of our DBFC-Net framework can be downloaded from (Extraction code 1u1f) [Baidu Cloud](https://pan.baidu.com/s/14_XWs5tR53KKG2hUHKadFQ).

How to use
---
The code is currently tested only on GPU.
- Source Code

   Prepare audio data
    ```
    python audio.py
    ```

  Training
   ```
   python train.py
   ```

  Testing
   ```
   python test.py
   ```
