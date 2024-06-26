# Dynamic_Graph_Representation
This is the official code of AAAI paper [《Dynamic Graph Representation for Occlusion Handling in Biometrics》](https://arxiv.org/abs/1912.00377v2)
  
We propose a novel unified framework integrated the merits of both CNNs and graphical models to learn dynamic graph representations for biometrics ercognition, called Dynamic Graph Representation (DGR). Convolutional features onto certain regions are re-crafted by a graph generator to establish the connections among the spatial parts of biometrics and build Feature Graphs based on these node representations. Each node of Feature Graphs corresponds to a specific part of the input image and the edges express the spatial relationships between parts. By analyzing the similarities between the nodes, the framework is able to adaptively remove the nodes representing the occluded parts. During dynamic graph matching, we propose a novel strategy to measure the distances of both nodes and adjacent matrixes. In this way, the proposed method is more convincing than CNNs-based methods because the dynamic graph method implies a more illustrative and reasonable inference of the biometrics decision.

![arch](main_idea_7.png)

## The proposed framework

![arch](framework.png)

# Usage Instructions

## Requirments

python == 3.7

pytorch == 1.1.0

torchvision == 0.3.0

## Training

### Data preparing

The recognition model is trained by normalized iris images. All of your training images should be stored in one folder. The labels should be recorded by a `.txt` file.
Image names are followed by labels in the label file. One row per image.

An example of label file:

![arch](txt_example.PNG)

### Start Training

`configs/config_train_singlescale.py`to set the configurations of training.

`train_singlescale.py` to begin training.

## Feature extraction

Pretrained model can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1fOkdHLThw5x0QiFT2f5zYQ), code: g3sn
or [Google Drive](https://drive.google.com/drive/folders/1It2f9mif3zdQJmu83LgFMCTwo9sAm8Od?usp=sharing)

`configs/config_FE.py`to set the configurations of feature extraction.

`feature_extraction_singlescale.py` to extract features.

## Test

`configs/config_test.py`to set the configurations of test.

`test.py` to test.

### Performance

#### iris recognition
Dataset       | ND-LG4000 | CASIA-Distance | CASIA-M1S2 | CASIA-Lamp
--- |--- |--- |--- |--- 
FRR@FAR=0.01% | 3.02%     | 6.94%          | 6.57       | 5.92%
EER           | 0.62%     | 1.71%          | 0.76%      | 0.61%

#### face recognition
![arch](face_recognition.PNG)

The pretrained model of face recognition is not available because of intellectual property policies.
But you can train your own model according to our codes.

# Update

Multi-scale strategy is integrated into the DGR. The representations of nodes from one single layer can only ingest contexts from receptive fields of the same size. Thus, the multiscale strategy is further incorporated to attain more diverse nodes representing regions of various sizes. The primitive FG is subsequently reorganized in a hierarchical manner for escalated DGM. Feature graphs are generated from feature maps of different layers. Multiscale content representations and topological structures, which are contained in multiscale feature graphs, are summarized together in the framework of our multiscale dynamic graph representation. Node features yielded from different layers correspond to different scales of local regions of the input image. Edges from different layers represent topological structures of different scales. Hence, the features contained in multiscale feature graphs representation are much more abundant than those in single-scale representation.

![arch](framework_multiscale.png)

The **training**, **feature extraction** and **test** are similar with its the single-scale counterpart.

# Citation
If you find **DGR** useful in your research, please consider to cite:

    @inproceedings{ren2020dynamic,
      title={Dynamic Graph Representation for Occlusion Handling in Biometrics.},
      author={Ren, Min and Wang, Yunlong and Sun, Zhenan and Tan, Tieniu},
      booktitle={AAAI},
      pages={11940--11947},
      year={2020}
    }

    @ARTICLE{10193782,
      author={Ren, Min and Wang, Yunlong and Zhu, Yuhao and Zhang, Kunbo and Sun, Zhenan},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
      title={Multiscale Dynamic Graph Representation for Biometric Recognition with Occlusions}, 
      year={2023},
      volume={},
      number={},
      pages={1-17},
      doi={10.1109/TPAMI.2023.3298836}}


