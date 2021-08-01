#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: tetralith_script_1d_resnet152.py
Date: 31/07/2021
Author: Lennart Hohmann
Description: This python script was used to train the model on the Tetralith HPC cluster of
             the Swedish National Supercomputer Centre (NCS). All code contained in this
             script is also found in the notebook file 1D_ResNet.ipynb, which contains additional
             lines of code to interactively check whether certain steps were correctly executed.
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from fastai import *
from fastai.vision.all import *
from sklearn.preprocessing import OneHotEncoder

"""# 2. Define functions required for creating the Datablock"""

# defining transform functions required for correctly loading the data when building the DataBlock

# one hot encoding function
def OH_enc(seq: str):
    # get the categories into array
    cats = ['K', 'D', 'N', 'E', 'R', 'A', 'T', 'L', 'I', 'Q', 'C', 'F', 'G', 'W', 'M', 'S', 'H', 'P', 'V', 'Y']
    cat_array = np.array(sorted(cats), ndmin=1) #
    # get seq into array
    trunc_seq = seq[:300] # truncate sequences longer than 300 
    seq_array = np.array(list(trunc_seq))
    # one hot encode the sequence
    onehot_encoder = OneHotEncoder(categories=[cat_array],sparse=False,handle_unknown='ignore')
    onehot_encoded_seq = onehot_encoder.fit_transform(seq_array.reshape(len(seq_array), 1))
    return np.transpose(onehot_encoded_seq)

# zero padding function that makes sure the encoded sequences are all of the same format later
def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

# combine in one function
def main_item_tfms(seq): 
    enc_seq = OH_enc(seq)
    pad_encseq_array = pad_along_axis(enc_seq,300,1)
    return pad_encseq_array

# functions to get x and y data
def get_y(r): return r['Knum'] # y are the labels in the Knum column in the dataframe
def get_x(r): return main_item_tfms(r['Seq']) # x are the sequences in the Seq column; the transform functinos are applied

# defining the TransformBlock for single-label categorical targets to be used to apply additional transforms when building the datablock
def CategoryBlock(vocab=None, sort=True, add_na=False):
    return TransformBlock(type_tfms=Categorize(vocab=vocab, sort=sort, add_na=add_na))

# read in my data from which training and validation set will be created
trainval = pd.read_csv('data/trainval.csv', low_memory=False)

# building the datablock, which acts as a template on how to load the data
dblock = DataBlock(blocks=(TransformBlock(batch_tfms=IntToFloatTensor), CategoryBlock()),
                   splitter = TrainTestSplitter(test_size=0.2, random_state=42, stratify=trainval[['Knum']]), # stratified split the trainval df into 80% train and 20% valid set
                   get_x = get_x,
                   get_y = get_y)

"""# 3. Create the dataloader"""

# create dataloaders from datablock 
dls = dblock.dataloaders(trainval, bs=256, shuffle=True, drop_last=True) # shuffle the data to prevent overfitting due to an organized dataset and drop the last incomplete batch; chose batch size = 256

# access the dataset from datablock creation template
dsets = dblock.datasets(trainval)

# label distribution in train set
train_label_count = Counter(dsets.train.items.Knum)

# calc. weights for each label class
class_weights = {} # empty dict to be filled with the class weights
for label in train_label_count:
    class_weights[label] = 1/train_label_count[label] # for every category the weight is (1 / number of associated sequences)
wgts = dsets.train.items.Knum.map(class_weights).values[:len(dsets.train)] 
weighted_dls = dblock.dataloaders(trainval,bs=256, dl_type=WeightedDL, wgts=wgts, shuffle=True, drop_last=True) 
dls.train = weighted_dls.train # replace the train dl with a weighted dl
# in the end the train dls is balanced and the valid dls is unbalanced -> later a suitable metric has to be chosen

"""# 4. Functions for the Learner"""

# the metric 
# choose one that is also fine for unbalanced datasets such as the F1score
F1Score = F1Score(average='macro') # 'macro' calculate score for each label individually, and then find their unweighted mean. penalizes poor performance on minority classes more

# callback to change the tensor type of the input to match the model weights type
class TensorTypeChange(Callback): 
    def before_batch(self):
        new_xb = [x.type(torch.FloatTensor).cuda() for x in self.learn.xb]
        self.learn.xb = new_xb
        return self.learn.xb

# callback to save the model during training
smc = SaveModelCallback(monitor="f1_score", fname="1D_ResNet152", comp=np.greater, with_opt=True) # change comp based on metric

# callback to stop training after x epochs when the model doesnt improve anymore
estop = EarlyStoppingCallback(monitor="f1_score", comp=np.greater, patience=15) # note: change comp based on chosen metric

# function to get the learner
def get_learner(m): # .to_fp16()
    return Learner(dls, m, loss_func=nn.CrossEntropyLoss(), opt_func=Adam, lr=defaults.lr, cbs=[TensorTypeChange,smc,estop], metrics=[accuracy,F1Score], model_dir='models')

"""# Bottleneck resnet: Book-based architecture"""

# the resnet stem
def _resnet_stem(*sizes): 
    return [
        ConvLayer(sizes[i], sizes[i+1], 3, stride = 2, ndim=1 if i==0 else 1) for i in range(len(sizes)-1) # ndim
    ] + [nn.MaxPool1d(kernel_size=3, stride=2, padding=1)] # 1d

# the conv block definition and ResBlock
def _conv_block(ni,nf,stride): 
    return nn.Sequential(
        ConvLayer(ni, nf//4, 1, ndim=1), # ndim
        ConvLayer(nf//4, nf//4, stride=stride, ndim=1), #
        ConvLayer(nf//4, nf, 1, act_cls=None, norm_type=NormType.BatchZero, ndim=1)) #

class ResBlock(Module):
    def __init__(self, ni, nf, stride=1):
        self.convs = _conv_block(ni,nf,stride)
        self.idconv = noop if ni==nf else ConvLayer(ni, nf, 1, act_cls=None, ndim=1) # ndim
        self.pool = noop if stride==1 else nn.AvgPool1d(2, ceil_mode=True) # changed to 1d
    
    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))

# putting it all together
class ResNet(nn.Sequential):
    def __init__(self, n_out, layers, expansion=1):
        stem = _resnet_stem(20,32,32,64) # 
        self.block_szs = [64, 64, 128, 256, 512]
        for i in range(1,5): self.block_szs[i] *= expansion 
        blocks = [self._make_layer(*o) for o in enumerate(layers)] 
        super().__init__(*stem, *blocks,
                        nn.AdaptiveAvgPool1d(1), Flatten(), #1d
                        nn.Linear(self.block_szs[-1], n_out))

    def _make_layer(self, idx, n_layers): 
        stride = 1 if idx==0 else 2
        ch_in,ch_out = self.block_szs[idx:idx+2] 
        return nn.Sequential(*[
            ResBlock(ch_in if i==0 else ch_out, ch_out, stride if i==0 else 1)
            for i in range(n_layers) 
        ])

# defining the architecture
rn = ResNet(dls.c, [3, 8, 36, 3], 4) # this would be a resnet152
# get the learner
learn = get_learner(rn)

"""# 6. Bottleneck resnet: Training"""

# training the model
epochs=200  # For the training on the Tetralith cluster this is set to 200
lr=0.00010964782268274575 # based on valley in lr_find 
learn.fit_one_cycle(epochs, lr) 

# call learn.export to save all the information of our Learner object for inference
# SaveModelCallback loads the best model at the end so the best model is exported
learn.export(fname='export.pkl') 

"""# 7. Model evaluation on testset"""

# slightly change the TensorTypeCallback so that the input is no longer put on gpu (as the weights arent either here)
# removed the .cuda() 
class TensorTypeChange(Callback): 
    def before_batch(self):
        new_xb = [x.type(torch.FloatTensor) for x in self.learn.xb] # removed the .cuda() 
        self.learn.xb = new_xb
        return self.learn.xb

# load the learner containing the template on how to load data
learn_inf = load_learner('export.pkl')

# remove non required callbacks
learn_inf.remove_cbs([SaveModelCallback,EarlyStoppingCallback])

# read in the test dataset that was previously created
test_df = pd.read_csv('data/test.csv', low_memory=False)

# create test dataloader
test_dl = learn_inf.dls.test_dl(test_df, with_labels=True) # based on dblock template

# Validate on dl
validation_data = learn_inf.validate(dl=test_dl)
# print the test fscore
print("Test set Fscore:",validation_data[2])

"""# 8. Test data set confusion matrix"""

# create dataloaders with twice the test_dl (only one will be used later for the CM)
dls = DataLoaders(test_dl,test_dl)

# to plot the construction matrix, the learner has to be constructed without the loss function (otherwise it gives an assertion error)
def get_learner(m):
    return Learner(dls, m, opt_func=Adam, lr=defaults.lr, cbs=[TensorTypeChange,smc], metrics=[accuracy,F1Score], model_dir='models') # no loss function provided (unclear on exact cause of error)

# architecture
rn = ResNet(dls.c, [3, 8, 36, 3], 4) # resnet152
learn = get_learner(rn)

# load the model from the /model directory which was created automatically during the training
learn = learn.load('1D_ResNet152') # the model you want to load

# switch out the dataloaders to get the CM on the test set
learn.dls = dls

# interpreting the classification results of the model
interp = ClassificationInterpretation.from_learner(learn)

# calculate the accuracy per category
confusion_data = interp.confusion_matrix()
cm = confusion_data.astype('float') / confusion_data.sum(axis=1)[:, np.newaxis]
accuracies = cm.diagonal()
for i in range(dls.c):
    if i >= 1:
        print(dls.vocab[i]," : ", round(accuracies[i],2))
    else:
        print(dls.vocab[0]," : ", round(accuracies[0],2))

# get the confusion matrix
interp.plot_confusion_matrix(figsize=(26,26), dpi=60)
plt.savefig('test_confusion_matrix.pdf',format='pdf')

