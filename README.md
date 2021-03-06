### [Pro/Con: Neural Detection of Stance in Argumentative Opinions](https://easychair.org/publications/preprint/4VQX)

Accurate information from both sides of the contemporary issues is known to be an `antidote in confirmation bias'.  While these types of information help the educators to improve their vital skills including critical thinking and open-mindedness, they are relatively rare and hard to find online. With the well-researched argumentative opinions (arguments) on controversial issues shared by Procon.org in a nonpartisan format, detecting the stance of arguments is a crucial step to automate organizing such resources. We use a universal pretrained language model with weight-dropped LSTM neural network to leverage the context of an argument for stance detection on the proposed dataset. Experimental results show that the dataset is challenging, however, utilizing the pretrained language model fine-tuned on context information yields a general model that beats the competitive baselines. We also provide analysis to find the informative segments of an argument to our stance detection model and investigate the relationship between the sentiment of an argument with its stance.


The paper is accepeted at SBP-BRiMS 2019. [Preprint version](https://easychair.org/publications/preprint/4VQX), [Springer version](https://link.springer.com/chapter/10.1007/978-3-030-21741-9_3)


### Dataset: 

**The procon dataset is purely for academic/research use and not for commercial purposes. Rights to the data belong to [procon.org](http://procon.org/) as they hosted the data.**


Please contact me (![email:](https://raw.githubusercontent.com/marjanhs/stance/master/email.png)) to request the dataset!

## Requirements

-fastai 2018 release ([version 1-.0.6](https://github.com/fastai/fastai/tree/release-1.0.6) or later in [2018](https://github.com/fastai/fastai/branches))

-nltk

-PyTorch


### Preprocessing

`procon_ai_utils.py` converts the dataset files to token and  token-id files and creates the itos/stoi files.

### Model

`__main__.py` fine-tunes pre-trained language model by fastai, trains and evalates the model
