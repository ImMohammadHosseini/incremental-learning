## :zap:A Novel incremental learning method for text classification:zap:
<p> Tensorflow implementation of: </p>

[![DOI:10.1016/j.eswa.2020.113198](http://img.shields.io/badge/DOI-10.1016/j.eswa.2020.113198-1589F0.svg)](https://doi.org/10.1016/j.eswa.2020.113198)

:fire: Text classification based on incremental learning :fire:

### :bookmark: introduction

[![-----------------------------------------------------]( 
https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)](https://github.com/ImMohammadHosseini/incremental-learning?tab=repositories)

 limitation of  text classification using deep learning:
1. deep learning models are trained in a batch learning setting with the entire training data. So it is cumbersome when used in a continual learning setting because storing previous text data is a memory expensive task
1. Second, it is difficult to obtain total sufficient labeled samples for text classification with deep learning at the beginning because the labeling cost is prohibitively expensive or they do not occur frequently enough to be collected
1. model performances highly depend on the distribution of the data samples. The distribution of the previous datasets might differ from that of the newly collected data, which might lead to overfitting.

to address thesee problems incremental learning is used.

benefits of inceremental learning:
1.when new samples are added, there is no need to retrain with all samples.

![The structure](images/1.png)

### :bookmark: method

[![-----------------------------------------------------]( 
https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)](https://github.com/ImMohammadHosseini/incremental-learning?tab=repositories)

this project consists of four components: Student model, a reinforcement learning (RL) module, a Teacher model, and a discriminator model.

#### Student model
Student models are deep learning models that are used to solve classification tasks. In this project, the pre-trained Bert model was utilized, followed by feeding the inputs to multilayer LSTM models. The input categories were then classified using softmax activation functions. The feature vectors of the text representations are captured in the vector of the last layer.

#### Reinforcement Learning Module

#### Teacher Model

#### Discriminator Modelll
