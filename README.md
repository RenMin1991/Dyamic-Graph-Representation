# Dyamic_Graph_Representation
This is the code of AAAI paper 《Dynamic Graph Representation for Occlusion Handling in Biometrics》
  
We propose a novel unified framework integrated the merits of both CNNs and graphical models to learn dynamic graph representations for biometrics ercognition, called Dynamic Graph Representation (DGR). Convolutional features onto certain regions are re-crafted by a graph generator to establish the connections among the spatial parts of biometrics and build Feature Graphs based on these node representations. Each node of Feature Graphs corresponds to a specific part of the input image and the edges express the spatial relationships between parts. By analyzing the similarities between the nodes, the framework is able to adaptively remove the nodes representing the occluded parts. During dynamic graph matching, we propose a novel strategy to measure the distances of both nodes and adjacent matrixes. In this way, the proposed method is more convincing than CNNs-based methods because the dynamic graph method implies a more illustrative and reasonable inference of the biometrics decision.

![arch](main_idea_7.png)

## Usage Instructions

### Requirments

python == 3.7

pytorch == 1.1.0

torchvision == 0.3.0
