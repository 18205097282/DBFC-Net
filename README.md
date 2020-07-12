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
   * Put the audio dataset to the ```audio_dataset``` folder.
    ```
    python audio.py
    ```

  Training
   * Download dataset to the ```dataset``` folder.
   * In ```main.py```  
     >* modify ```model_path``` to the path where you want to save your parameters of networks.
     >*  modify ```lr in params``` to ```0.001```, ```momentum in paramss```  to ```0.9```.  
     >*  modify ```step_siz in StepLR``` to ```5```, ```gammam in StepLR```  to ```0.8```.    
   ```
   python main.py
   ```

  Testing
     * If you just want to do a quick test on the model and check the final retrieval performance, please follow the subsequent steps.
     Download dataset to the ```dataset``` folder.
     * Download the trained models of our work  and put it to the  ```models``` folder.
   ``` 
   python test.py
   ```
