# fixed target location

# In[82]:
winSize = True

# coding: utf-8
print('creating design...')

# In[2]:


import numpy as np
import pandas as pd
import os, glob
import csv
import copy

# # Install a pip package in the current Jupyter kernel
# import sys
# !{sys.executable} -m pip install psychopy
from psychopy import visual,core,gui,event
import pandas as pd
import numpy as np
import random
import sys
import csv
from psychopy.tools.monitorunittools import posToPix 
from psychopy import parallel
from psychopy import event
import datetime

# In[83]:


# subjID='asdfghjkl'
myDlg = gui.Dlg(title="PrAff")
myDlg.addText('Please enter your subject ID')
myDlg.addField('Subj ID:')
myDlg.addText('Total session number')
myDlg.addField('Number of sessions:')
myDlg.addText('Session number')
myDlg.addField('Session:')
myDlg.addText('Is this an EEG session?')
myDlg.addField('eeg:')
subjDlg = myDlg.show() # show dialog and wait for OK or Cancel
subjID = subjDlg[0]
num_sessions = int(subjDlg[1])
sessionNum = int(subjDlg[2])
eeg = int(subjDlg[3])
if myDlg.OK:  # or if ok_data is not None
    print('subj:',subjID)
    print('total sessions:',num_sessions)
    print('session:',sessionNum)
    print('eeg:',eeg)
else:
    print('user cancelled')
    
folder = {'root': 'C:/Users/iaslra/Desktop/PrAff_EEG/'}
# folder = {'root': '/Users/chendanlei/Desktop/PrAff_EEG/'}
os.chdir(folder['root'])
# print(os.getcwd())

folder['code_path']= folder['root']+'code/'
folder['stim']= folder['root']+'stimuli/'
folder['negstim']= folder['stim']+'negative'
folder['neustim']= folder['stim']+'neutral'
folder['sample']= folder['stim']+'sample'
folder['scale']= folder['stim']+'scale/'
folder['output'] = folder['root']+'output'
folder['design_output'] = folder['root']+'design_output'


# In[84]:


### Parameters (Input)
#from item{i}.attribute1(2,j) to item[i-1]['attribute1'][2,j-1]

#one is 2&3 =180trials_per_block, one is 3&2 =168trials_per_block
param ={'num_examplars_structured': 6}
param['NN_pairs'] = param['num_examplars_structured']
param['NB_pairs'] = param['num_examplars_structured']
param['BN_pairs'] = param['num_examplars_structured']
param['BB_pairs'] = param['num_examplars_structured']
param['num_examplars_unstructured']=1
param['NN_rand_pairs'] = param['num_examplars_unstructured']
param['BN_rand_pairs'] = param['num_examplars_unstructured']+param['num_examplars_unstructured']
param['BB_rand_pairs'] = param['num_examplars_unstructured']
param['total_rand_pairs'] = param['NN_rand_pairs']+param['BN_rand_pairs']+param['BB_rand_pairs']
param['pair_types'] = 7
param['rep_per_block'] = 3
param['num_blocks'] = 6
param['num_sessions'] = num_sessions

# In[5]:

param['total_pairs']=param['NN_pairs']+param['NB_pairs']+param['BN_pairs']+param['BB_pairs']+param['NN_rand_pairs']+param['BN_rand_pairs']+param['BB_rand_pairs']
param['total_structured_pairs']=param['NN_pairs']+param['NB_pairs']+param['BN_pairs']+param['BB_pairs']
param['total_unstructured_pairs']=param['NN_rand_pairs']+param['BN_rand_pairs']+param['BB_rand_pairs']
param['total_structured_stimuli']=2*param['total_structured_pairs']
param['total_unstructured_stimuli']=2*param['total_unstructured_pairs']
param['total_stimuli']=2*param['total_pairs']
param['total_negstim']=param['NN_pairs']*0+param['NB_pairs']*1+param['BN_pairs']*1+param['BB_pairs']*2+param['NN_rand_pairs']*0+param['BN_rand_pairs']*1+param['BB_rand_pairs']*2
param['total_neustim']=param['NN_pairs']*2+param['NB_pairs']*1+param['BN_pairs']*1+param['BB_pairs']*0+param['NN_rand_pairs']*2+param['BN_rand_pairs']*1+param['BB_rand_pairs']*0

param['trials_per_block']= param['total_stimuli'] * param['rep_per_block']
param['total_num_rep'] = param['rep_per_block']*param['num_blocks']

param['perc_greypatch'] = 1; #percentage of images that contains gray scale patch
param['num_greystimuli'] = int(round(param['total_stimuli']*param['perc_greypatch']));
param['num_nongreystimuli'] = param['total_stimuli']-param['num_greystimuli'];
param['num_greyimages'] = int(round(param['trials_per_block']*param['perc_greypatch']));
param['num_nongreyimages'] = param['trials_per_block']-param['num_greyimages'];

param['display_time'] = 1;
param['fixation_time'] = 0.5;
total_time = (param['total_stimuli']*(param['display_time']+param['fixation_time'])*param['rep_per_block']+30)*param['num_blocks']/60

# EEG marker
# param['stimID'] = [[1,2],[11,12],[21,22],[31,32],[41,42],[51,52],[3,4],[13,14],[23,24],[33,34],[43,44],[53,54],[5,6],[15,16],[25,26],[35,36],[45,46],[55,56],[7,8],[17,18],[27,28],[37,38],[47,48],[57,58],[61,62],[65,66],[67,68]]
# param['stimPairTypeID'] = [[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[3,4],[3,4],[3,4],[3,4],[3,4],[3,4],[5,6],[5,6],[5,6],[5,6],[5,6],[5,6],[7,8],[7,8],[7,8],[7,8],[7,8],[7,8],[61,62],[63,64],[63,64],[65,66]]
param['stimID'] = [[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[3,4],[3,4],[3,4],[3,4],[3,4],[3,4],[5,6],[5,6],[5,6],[5,6],[5,6],[5,6],[7,8],[7,8],[7,8],[7,8],[7,8],[7,8],[101,102],[105,106],[105,106],[107,108]]
param['correctResponseMarker'] = 100
param['incorrectResponseMarker'] = 200
param['breakMarker'] = 10
param['fixationMarker'] = 70
param['neuFoilImageMarker'] = 80
param['negFoilImageMarker'] = 80
param['mutualResponseMarker'] = 90

# Other parameters are loaded from the design file.

#image dimensions
param['fixation_width']=5
param['fixation_location']=((0, -15), (0, 15), (0,0), (-15,0), (15, 0))   

win = visual.Window(fullscr = winSize)
param['img_y'] = 1
param['img_x'] = win.size[1]*(param['img_y']/win.size[0])/480*640
param['inverted_patch_loc'] = [win.size[1]*(-0.05/win.size[0]), 0.05, 
                               win.size[1]*(0.05/win.size[0]), -0.05]
param['patch_length']=[win.size[1]*(0.05/win.size[0])*2, 0.05*2]
win.close()

# timing
param['stim_time']= 1.5
param['fixation_time']=0.5
param['breakTime']=15

ok_response = 'j','q','enter','f','k'
correct_key = 'j'
cancel_key = 'q'
loop_instruction_key = 'f'

# In[6]:

## Collect Stimuli.
# Negative stimuli
os.chdir(folder['negstim'])
negstim = glob.glob('*.jpg')#glob.glob returns a list of all files ends in .jpg, * means wildcards
for n in range(len(negstim)):
  negstim[n] = folder['negstim']+'/'+negstim[n]

# Neutral stimuli
os.chdir(folder['neustim'])
neustim = glob.glob('*.jpg')#glob.glob returns a list of all files ends in .jpg, * means wildcards
for n in range(len(neustim)):
  neustim[n] = folder['neustim']+'/'+neustim[n]

# Resize the struct to match the num of stimuli
if len(negstim) > param['total_negstim']:
    randperm=[negstim[i] for i in np.random.permutation(len(negstim))]
  # make equal number of stimuli needed
    negstim=randperm[0:param['total_negstim']] 
    negstim_foil = randperm[param['total_negstim']:param['total_negstim']+param['total_negstim']]
if len(neustim) > param['total_neustim']:
    randperm=[neustim[i] for i in np.random.permutation(len(neustim))]
    neustim=randperm[0:param['total_neustim']]
    neustim_foil=randperm[param['total_neustim']:param['total_neustim']+param['total_neustim']]

# Stimuli shuffled randomly for each subject.
stimuli={'neg': negstim}
stimuli['neg_foil']=negstim_foil
stimuli['neg']=[stimuli['neg'][i] for i in np.random.permutation(len(stimuli['neg']))]
stimuli['neg_foil']=[stimuli['neg_foil'][i] for i in np.random.permutation(len(stimuli['neg_foil']))]
stimuli['neu']= neustim
stimuli['neu_foil']= neustim_foil
stimuli['neu']=[stimuli['neu'][i] for i in np.random.permutation(len(stimuli['neu']))]
stimuli['neu_foil']=[stimuli['neu_foil'][i] for i in np.random.permutation(len(stimuli['neu_foil']))]

del negstim,neustim,negstim_foil,neustim_foil

def findIndexIn3D(list3D, number):
    #list3D has to be a 3D list
    #number is the value you want to find in the list
    #the function returns the index of the 3D list given the value of number
    indexList = []
    for index in range(len(list3D[0])):
#         print('index:', index, list3D[0][index])
        indexMatrix = np.argwhere(np.array(list3D[0][index]) == number)
#         print(indexMatrix)
        indexList.append( (index, indexMatrix) )
    return indexList

def giveNewValueIn3D(indexList, newValue, newList3D):
    #newValue is a 1D list
    #size and type of newList3D has to be predefined
    #the function returns the new 3D list after placing the new value in the list of indices
    count = 0
    for eachMatrix in indexList:
        matrixIndex = eachMatrix[0]
        for eachIndex in eachMatrix[1]:
#             print ('%s, index1: %s, index2: %s, yIndex: %s'%(matrixIndex, eachIndex[0], eachIndex[1], count))
            path = newValue[count]         
#             print ('picPath: %s'%path)
#             print(newList3D[matrixIndex][eachIndex[0]][eachIndex[1]])
#             print(matrixIndex,eachIndex[0],eachIndex[1])
            newList3D[matrixIndex][eachIndex[0]][eachIndex[1]] = path
            count += 1
#             print(newList3D[matrixIndex][eachIndex[0]][eachIndex[1]])
#             print(newList3D)
    return newList3D

#this def rearrange a multidimensional list to a scalar-like list... not generic, use with caution
#this function is used here as a better way to 
def reshape3DToScalar(originalList,newList):
    for x in range(len(originalList)):
#         print('x')
#         print(originalList[x])
        for y in range(len(originalList[x])):
#             print('y')
#             print(originalList[x][y])
            for z in range(len(originalList[x][y])):
#                 print('z')
#                 print(originalList[x][y][z])
                newList.append(originalList[x][y][z])
#     print(len(newList))
    return newList

#this def rearrange a multidimensional list to a scalar-like list... not generic, use with caution
#this function is used here as a better way to 
def reshape2DToScalar(originalList,newList):
    for x in range(len(originalList)):
#         print('x')
#         print(originalList[x])
        for y in range(len(originalList[x])):
#             print('y')
#             print(originalList[x][y])
            newList.append(originalList[x][y])
#     print(len(newList))
    return newList

#this def rearrange a multidimensional list to a scalar-like list... not generic, use with caution
#this function is used here as a better way to 
def reshape4DToScalar(originalList,newList):
    for x in range(len(originalList)):
#         print('x')
#         print(originalList[x])
        for y in range(len(originalList[x])):
#             print('y')
#             print(originalList[x][y])
            for z in range(len(originalList[x][y])):
#                 print('z')
#                 print(originalList[x][y][z])
                for a in range(len(originalList[x][y][z])):
                    newList.append(originalList[x][y][z][a])
#     print(len(newList))
    return newList


# In[85]:
if sessionNum == 0:
    
    # In[9]:


    ## Create Stimuli/Condition Order

    # Build default block.
    # Condition:
    # Affective = 1 % Neutral = 0
    design={}
    design['base']={}
    
    design['base']['stimID'] = param['stimID']
#     design['base']['stimPairTypeID'] = param['stimPairTypeID']

    # initialie empty list
    # note that we can't use ['','','']*n here since the pointer will assign value to each element accordingly
    design['base']['condition'] = []
    for i in range(param['pair_types']):
        design['base']['condition'].append([])
    for i in range(param['NN_pairs']):
        design['base']['condition'][0].append([])
        design['base']['condition'][0][i]=[0,0]
    for i in range(param['NB_pairs']):
        design['base']['condition'][1].append([])
        design['base']['condition'][1][i]=[0,1]
    for i in range(param['BN_pairs']):
        design['base']['condition'][2].append([])
        design['base']['condition'][2][i]=[1,0]
    for i in range(param['BB_pairs']):
        design['base']['condition'][3].append([])
        design['base']['condition'][3][i]=[1,1]
    for i in range(param['NN_rand_pairs']):
        design['base']['condition'][4].append([])
        design['base']['condition'][4][i]=[0,0]
    for i in range(param['BN_rand_pairs']):
        design['base']['condition'][5].append([])
        design['base']['condition'][5][i]=[1,0]
    for i in range(param['BB_rand_pairs']):
        design['base']['condition'][6].append([])
        design['base']['condition'][6][i]=[1,1]
    design['base']['condition']=[np.array(design['base']['condition'])] # has to be nparray to use np.where in the function
    
    design['base']['stimuli']=[]
    for i in range(param['pair_types']):
        design['base']['stimuli'].append([])
    for i in range(param['NN_pairs']):
        design['base']['stimuli'][0].append([])
        design['base']['stimuli'][0][i]=['','']
    for i in range(param['NB_pairs']):
        design['base']['stimuli'][1].append([])
        design['base']['stimuli'][1][i]=['','']
    for i in range(param['BN_pairs']):
        design['base']['stimuli'][2].append([])
        design['base']['stimuli'][2][i]=['','']
    for i in range(param['BB_pairs']):
        design['base']['stimuli'][3].append([])
        design['base']['stimuli'][3][i]=['','']
    for i in range(param['NN_rand_pairs']):
        design['base']['stimuli'][4].append([])
        design['base']['stimuli'][4][i]=['','']
    for i in range(param['BN_rand_pairs']):
        design['base']['stimuli'][5].append([])
        design['base']['stimuli'][5][i]=['','']
    for i in range(param['BB_rand_pairs']):
        design['base']['stimuli'][6].append([])
        design['base']['stimuli'][6][i]=['','']
    design['base']['greypatch_positionx'] = copy.deepcopy(design['base']['stimuli'])
    design['base']['greypatch_positiony'] = copy.deepcopy(design['base']['stimuli'])
    design['base']['greypatch_positionLorR'] = copy.deepcopy(design['base']['stimuli'])

    # assign stimuli: Affective = 1  Neutral = 0
    indexList1 = findIndexIn3D(design['base']['condition'], 1)
    indexList0 = findIndexIn3D(design['base']['condition'], 0)

    design['base']['stimuli'] = giveNewValueIn3D(indexList1, stimuli['neg'], design['base']['stimuli'])
    design['base']['stimuli'] = giveNewValueIn3D(indexList0, stimuli['neu'], design['base']['stimuli'])
    design['base']['stimuli']=[np.array(design['base']['stimuli'])] #to keep all ele in dict the same structure
    # design['base']['condition']=design['base']['condition'][0].tolist()

    greyp_patch_LorR_neg = ['L','R']*int(len(stimuli['neg'])/2)
    if len(stimuli['neg']) %2 != 0:
        greyp_patch_LorR_neg.extend(greyp_patch_LorR_neg[len(greyp_patch_LorR_neg)-1])
    randomizer = np.random.permutation(len(greyp_patch_LorR_neg))
    greyp_patch_LorR_neg = [greyp_patch_LorR_neg[i] for i in randomizer]
    design['base']['greypatch_positionLorR'] = giveNewValueIn3D(indexList1, greyp_patch_LorR_neg, design['base']['greypatch_positionLorR'])
    
    grey_x=[]
    grey_y=[]
    for v in range(len(greyp_patch_LorR_neg)):
        x=param['img_x']/2
        y=param['img_y']/2
        if greyp_patch_LorR_neg[v] == 'L':
            #left
            x=random.uniform(-(param['img_x']/2)+param['patch_length'][0],0-param['patch_length'][1])
        elif greyp_patch_LorR_neg[v] == 'R':
            #right
            x=random.uniform(0+param['patch_length'][0],param['img_x']/2-param['patch_length'][1])
        y=random.uniform(-(param['img_y']/2)+param['patch_length'][0],param['img_y']/2-param['patch_length'][1])
        grey_x.append(x)
        grey_y.append(y) 
    design['base']['greypatch_positionx'] = giveNewValueIn3D(indexList1, grey_x, design['base']['greypatch_positionx'])
    design['base']['greypatch_positiony'] = giveNewValueIn3D(indexList1, grey_y, design['base']['greypatch_positiony'])

    greyp_patch_LorR_neu = ['L','R']*int(len(stimuli['neu'])/2)
    if len(stimuli['neu']) %2 != 0:
        greyp_patch_LorR_neu.extend(greyp_patch_LorR_neu[len(greyp_patch_LorR_neu)-1])
    greyp_patch_LorR_neu = [greyp_patch_LorR_neu[i] for i in np.random.permutation(len(greyp_patch_LorR_neu))]
    design['base']['greypatch_positionLorR'] = giveNewValueIn3D(indexList0, greyp_patch_LorR_neu, design['base']['greypatch_positionLorR'])
    
    grey_x=[]
    grey_y=[]
    for v in range(len(greyp_patch_LorR_neu)):
        x=param['img_x']/2
        y=param['img_y']/2
        if greyp_patch_LorR_neu[v] == 'L':
            #left
            x=random.uniform(-(param['img_x']/2)+param['patch_length'][0],0-param['patch_length'][1])
        elif greyp_patch_LorR_neu[v] == 'R':
            #right
            x=random.uniform(0+param['patch_length'][0],param['img_x']/2-param['patch_length'][1])
        y=random.uniform(-(param['img_y']/2)+param['patch_length'][0],param['img_y']/2-param['patch_length'][1])
        grey_x.append(x)
        grey_y.append(y) 
    design['base']['greypatch_positionx'] = giveNewValueIn3D(indexList0, grey_x, design['base']['greypatch_positionx'])
    design['base']['greypatch_positiony'] = giveNewValueIn3D(indexList0, grey_y, design['base']['greypatch_positiony'])

    design['base']['greypatch_positionx']=[np.array(design['base']['greypatch_positionx'])]
    design['base']['greypatch_positiony']=[np.array(design['base']['greypatch_positiony'])]
    design['base']['greypatch_positionLorR']=[np.array(design['base']['greypatch_positionLorR'])] #to keep all ele in dict the same structure

    design['base']['stimCond'] = []
    for i in range(param['pair_types']):
        design['base']['stimCond'].append([])
    for i in range(param['NN_pairs']):
        design['base']['stimCond'][0].append([])
        design['base']['stimCond'][0][i]=['Neu','Neu']
    for i in range(param['NB_pairs']):
        design['base']['stimCond'][1].append([])
        design['base']['stimCond'][1][i]=['Neu','Aff']
    for i in range(param['NB_pairs']):
        design['base']['stimCond'][2].append([])
        design['base']['stimCond'][2][i]=['Aff','Neu']
    for i in range(param['BB_pairs']):
        design['base']['stimCond'][3].append([])
        design['base']['stimCond'][3][i]=['Aff','Aff']
    for i in range(param['NN_rand_pairs']):
        design['base']['stimCond'][4].append([])
        design['base']['stimCond'][4][i]=['Neu','Neu']
    for i in range(param['BN_rand_pairs']):
        design['base']['stimCond'][5].append([])
        design['base']['stimCond'][5][i]=['Aff','Neu']
    for i in range(param['BB_rand_pairs']):
        design['base']['stimCond'][6].append([])
        design['base']['stimCond'][6][i]=['Aff','Aff']
    design['base']['stimCond']=[np.array(design['base']['stimCond'])] # has to be nparray to use np.where in the function

    # pairCond:
    design['base']['pairCond'] = []
    for i in range(param['pair_types']):
        design['base']['pairCond'].append([])
    for i in range(param['pair_types']):
        design['base']['pairCond'].append([])
    for i in range(param['NN_pairs']):
        design['base']['pairCond'][0].append([])
        design['base']['pairCond'][0][i]=[0,0]
    for i in range(param['NB_pairs']):
        design['base']['pairCond'][1].append([])
        design['base']['pairCond'][1][i]=[1,1]
    for i in range(param['BN_pairs']):
        design['base']['pairCond'][2].append([])
        design['base']['pairCond'][2][i]=[2,2]
    for i in range(param['BB_pairs']):
        design['base']['pairCond'][3].append([])
        design['base']['pairCond'][3][i]=[3,3]
    for i in range(param['NN_rand_pairs']):
        design['base']['pairCond'][4].append([])
        design['base']['pairCond'][4][i]=[4,4]
    for i in range(param['BN_rand_pairs']):
        design['base']['pairCond'][5].append([])
        design['base']['pairCond'][5][i]=[5,5]
    for i in range(param['BB_rand_pairs']):
        design['base']['pairCond'][6].append([])
        design['base']['pairCond'][6][i]=[6,6]
    design['base']['pairCond']=[np.array(design['base']['pairCond'])] # has to be nparray to use np.where in the function

    # assign cond names
    design['base']['pairCondN']=[]
    for i in range(param['pair_types']):
        design['base']['pairCondN'].append([])
    for i in range(param['NN_pairs']):
        design['base']['pairCondN'][0].append([])
        design['base']['pairCondN'][0][i]=['','']
    for i in range(param['NB_pairs']):
        design['base']['pairCondN'][1].append([])
        design['base']['pairCondN'][1][i]=['','']
    for i in range(param['BN_pairs']):
        design['base']['pairCondN'][2].append([])
        design['base']['pairCondN'][2][i]=['','']
    for i in range(param['BB_pairs']):
        design['base']['pairCondN'][3].append([])
        design['base']['pairCondN'][3][i]=['','']
    for i in range(param['NN_rand_pairs']):
        design['base']['pairCondN'][4].append([])
        design['base']['pairCondN'][4][i]=['','']
    for i in range(param['BN_rand_pairs']):
        design['base']['pairCondN'][5].append([])
        design['base']['pairCondN'][5][i]=['','']
    for i in range(param['BB_rand_pairs']):
        design['base']['pairCondN'][6].append([])
        design['base']['pairCondN'][6][i]=['','']

    indexList0 = findIndexIn3D(design['base']['pairCond'], 0)
    NeuNeu = ['NeuNeu','NeuNeu']*param['NN_pairs']
    design['base']['pairCondN'] = giveNewValueIn3D(indexList0, NeuNeu, design['base']['pairCondN'])
    indexList1 = findIndexIn3D(design['base']['pairCond'], 1)
    NeuAff = ['NeuAff','NeuAff']*param['NB_pairs']
    design['base']['pairCondN'] = giveNewValueIn3D(indexList1, NeuAff, design['base']['pairCondN'])
    indexList2 = findIndexIn3D(design['base']['pairCond'], 2)
    AffNeu = ['AffNeu','AffNeu']*param['BN_pairs']
    design['base']['pairCondN'] = giveNewValueIn3D(indexList2, AffNeu, design['base']['pairCondN'])
    indexList3 = findIndexIn3D(design['base']['pairCond'], 3)
    AffAff = ['AffAff','AffAff']*param['BB_pairs']
    design['base']['pairCondN'] = giveNewValueIn3D(indexList3, AffAff, design['base']['pairCondN'])
    indexList4 = findIndexIn3D(design['base']['pairCond'], 4)
    rand_NeuNeu = ['rand_NeuNeu','rand_NeuNeu']*param['NN_rand_pairs']
    design['base']['pairCondN'] = giveNewValueIn3D(indexList4, rand_NeuNeu, design['base']['pairCondN'])
    indexList5 = findIndexIn3D(design['base']['pairCond'], 5)
    rand_AffNeu = ['rand_AffNeu','rand_AffNeu']*param['BN_rand_pairs']
    design['base']['pairCondN'] = giveNewValueIn3D(indexList5, rand_AffNeu, design['base']['pairCondN'])
    indexList6 = findIndexIn3D(design['base']['pairCond'], 6)
    rand_AffAff = ['rand_AffAff','rand_AffAff']*param['BB_rand_pairs']
    design['base']['pairCondN'] = giveNewValueIn3D(indexList6, rand_AffAff, design['base']['pairCondN'])

    design['base']['pairCondN']=[np.array(design['base']['pairCondN'])] #to keep all ele in dict the same structure
    # design['base']['pairCond']=design['base']['pairCond'][0].tolist()

    design['base']['pairID']= [[0,0]]*param['total_pairs']
    for x in range(len(list(range(param['total_pairs'])))):
        design['base']['pairID'][x] = [list(range(param['total_pairs']))[x]]*2
    design['base']['position']= [[0,1]]*param['total_pairs']

    # In[111]:
    for s in range(param['num_sessions']):
        design['block']=[[]]*param['num_blocks']

        for n in range(param['num_blocks']):
            for m in range(param['rep_per_block']):
    #                 print('repetition:',m)  
                temp={};
                temp['block']={}
                #the following code is equivalent to matlab temp['block']['stimuli'] = temp['block']['stimuli'][temp['block_rand'][h]][:]
                temp['block']['stimuli']=[]
                temp['block']['stimuli']=reshape3DToScalar(design['base']['stimuli'],temp['block']['stimuli']) 
                temp['block']['condition']=[]
                temp['block']['condition']=reshape3DToScalar(design['base']['condition'],temp['block']['condition']) 
                temp['block']['pairCondN']=[]
                temp['block']['pairCondN']=reshape3DToScalar(design['base']['pairCondN'],temp['block']['pairCondN']) 
                temp['block']['pairCond']=[]
                temp['block']['pairCond']=reshape3DToScalar(design['base']['pairCond'],temp['block']['pairCond']) 
                temp['block']['stimCond']=[]
                temp['block']['stimCond']=reshape3DToScalar(design['base']['stimCond'],temp['block']['stimCond']) 
                temp['block']['greypatch_positionx']=[]
                temp['block']['greypatch_positionx']=reshape3DToScalar(design['base']['greypatch_positionx'],temp['block']['greypatch_positionx']) 
                temp['block']['greypatch_positiony']=[]
                temp['block']['greypatch_positiony']=reshape3DToScalar(design['base']['greypatch_positiony'],temp['block']['greypatch_positiony']) 
                temp['block']['greypatch_positionLorR']=[]
                temp['block']['greypatch_positionLorR']=reshape3DToScalar(design['base']['greypatch_positionLorR'],temp['block']['greypatch_positionLorR']) 
                temp['block']['pairID']=design['base']['pairID']
                temp['block']['position']=design['base']['position']
                temp['block']['stimID']=design['base']['stimID']

                #this is because we append the random pairlets from last
                temp['rand']={}
                for l in range(param['total_rand_pairs']): 
                    # randomize order within rand_pairlets.
                    temp['rand']['stim'] = temp['block']['stimuli'][-1-l]
                    temp['rand']['cond'] = temp['block']['condition'][-1-l]
                    temp['rand']['stimCond'] = temp['block']['stimCond'][-1-l]
                    temp['rand']['stimID'] = temp['block']['stimID'][-1-l]
                    temp['rand']['greypatch_positionx'] = temp['block']['greypatch_positionx'][-1-l]
                    temp['rand']['greypatch_positiony'] = temp['block']['greypatch_positiony'][-1-l]
                    temp['rand']['greypatch_positionLorR'] = temp['block']['greypatch_positionLorR'][-1-l]

                    rand_order = np.random.permutation(len(temp['rand']['stim']))
                    temp['block']['stimuli'][-1-l] = [temp['rand']['stim'][i] for i in rand_order]
                    temp['block']['condition'][-1-l] = [temp['rand']['cond'][i] for i in rand_order]
                    temp['block']['stimCond'][-1-l] = [temp['rand']['stimCond'][i] for i in rand_order]
                    temp['block']['stimID'][-1-l] = [temp['rand']['stimID'][i] for i in rand_order]
                    temp['block']['greypatch_positionx'][-1-l] = [temp['rand']['greypatch_positionx'][i] for i in rand_order]
                    temp['block']['greypatch_positiony'][-1-l] = [temp['rand']['greypatch_positiony'][i] for i in rand_order]
                    temp['block']['greypatch_positionLorR'][-1-l] = [temp['rand']['greypatch_positionLorR'][i] for i in rand_order]

                # randomize pairlet order across block.
                temp['block_rand'] = [list(range(len(temp['block']['stimuli'])))[i] for i in np.random.permutation(len(temp['block']['stimuli']))] # get randperm
                temp['block']['condition'] = [temp['block']['condition'][i] for i in temp['block_rand']]
                temp['block']['stimuli'] = [temp['block']['stimuli'][i] for i in temp['block_rand']]
                temp['block']['pairID'] = [temp['block']['pairID'][i] for i in temp['block_rand']]
                temp['block']['position'] = [temp['block']['position'][i] for i in temp['block_rand']]
                temp['block']['pairCondN'] = [temp['block']['pairCondN'][i] for i in temp['block_rand']]
                temp['block']['pairCond'] = [temp['block']['pairCond'][i] for i in temp['block_rand']]
                temp['block']['stimCond'] = [temp['block']['stimCond'][i] for i in temp['block_rand']]
                temp['block']['stimID'] = [temp['block']['stimID'][i] for i in temp['block_rand']]
                temp['block']['greypatch_positionx'] = [temp['block']['greypatch_positionx'][i] for i in temp['block_rand']]
                temp['block']['greypatch_positiony'] = [temp['block']['greypatch_positiony'][i] for i in temp['block_rand']]
                temp['block']['greypatch_positionLorR'] = [temp['block']['greypatch_positionLorR'][i] for i in temp['block_rand']]

                if m == 0:
                    design['block'][n]={}
                    design['block'][n]['stimuli'] = temp['block']['stimuli']
                    design['block'][n]['condition'] = temp['block']['condition']
                    design['block'][n]['pairID'] = temp['block']['pairID']
                    design['block'][n]['position'] = temp['block']['position']
                    design['block'][n]['pairCondN'] = temp['block']['pairCondN']
                    design['block'][n]['pairCond'] = temp['block']['pairCond']
                    design['block'][n]['greypatch_positionx'] = temp['block']['greypatch_positionx']
                    design['block'][n]['greypatch_positiony'] = temp['block']['greypatch_positiony']
                    design['block'][n]['greypatch_positionLorR'] = temp['block']['greypatch_positionLorR']
                    design['block'][n]['stimID'] = temp['block']['stimID']
                    #reshape stimCond into scalar
                    temp_stimCond = []
                    temp_stimCond = reshape2DToScalar(temp['block']['stimCond'],temp_stimCond) 
                    design['block'][n]['stimCond'] = temp_stimCond            
                else:
                    while temp['block']['stimuli'][0][0] == design['block'][n]['stimuli'][-1][0] or temp['block']['stimuli'][0][1] == design['block'][n]['stimuli'][-1][1]:

        #                 reshuffle the new block, as many times as it takes to
        #                 ensure that pairlet does not repeat across blocks (last 
        #                 pairlets of one block and first pairlet of the new block).
        #                 print('reshuffle')
                        temp['block_rand'] = [list(range(len(temp['block']['stimuli'])))[i] for i in np.random.permutation(len(temp['block']['stimuli']))] # get randperm
                        temp['block']['condition'] = [temp['block']['condition'][i] for i in temp['block_rand']]
                        temp['block']['stimuli'] = [temp['block']['stimuli'][i] for i in temp['block_rand']]
                        temp['block']['pairID'] = [temp['block']['pairID'][i] for i in temp['block_rand']]
                        temp['block']['position'] = [temp['block']['position'][i] for i in temp['block_rand']]
                        temp['block']['pairCondN'] = [temp['block']['pairCondN'][i] for i in temp['block_rand']]
                        temp['block']['pairCond'] = [temp['block']['pairCond'][i] for i in temp['block_rand']]
                        temp['block']['stimCond'] = [temp['block']['stimCond'][i] for i in temp['block_rand']]
                        temp['block']['stimID'] = [temp['block']['stimID'][i] for i in temp['block_rand']]
                        temp['block']['greypatch_positionx'] = [temp['block']['greypatch_positionx'][i] for i in temp['block_rand']]
                        temp['block']['greypatch_positiony'] = [temp['block']['greypatch_positiony'][i] for i in temp['block_rand']]
                        temp['block']['greypatch_positionLorR'] = [temp['block']['greypatch_positionLorR'][i] for i in temp['block_rand']]

        #             print('append')
        #              once there are no repeats, append to the existing block.
                    design['block'][n]['stimuli'].extend(temp['block']['stimuli'])
                    design['block'][n]['condition'].extend(temp['block']['condition'])
                    design['block'][n]['pairID'].extend(temp['block']['pairID'])
                    design['block'][n]['position'].extend(temp['block']['position'])
                    design['block'][n]['pairCondN'].extend(temp['block']['pairCondN'])
                    design['block'][n]['pairCond'].extend(temp['block']['pairCond'])
                    design['block'][n]['greypatch_positionx'].extend(temp['block']['greypatch_positionx'])
                    design['block'][n]['greypatch_positiony'].extend(temp['block']['greypatch_positiony'])
                    design['block'][n]['greypatch_positionLorR'].extend(temp['block']['greypatch_positionLorR'])
                    design['block'][n]['stimID'].extend(temp['block']['stimID'])
                    #reshape stimCond into scalar
                    temp_stimCond = []
                    temp_stimCond = reshape2DToScalar(temp['block']['stimCond'],temp_stimCond) 
                    design['block'][n]['stimCond'].extend(temp_stimCond)
                    design['block'][n]['greypatch'] = [1]*(param['num_greyimages']+param['num_nongreyimages'])
                del temp

            #to keep everything the same length (#of trials per block)
            for key in design['block'][n].keys():
                if len(design['block'][n][key]) != param['trials_per_block']:
                    temp=[]
                    temp=reshape2DToScalar(design['block'][n][key],temp) 
                    design['block'][n][key]=temp

            # In[ ]:
            # save block-wise file
            print('saving block ',n,' design data')
            os.chdir(folder['design_output'])
            fileName = subjID + '_session'+ str(s)+'_block' + str(n) + '_design.csv'
            df=pd.DataFrame.from_dict(design['block'][n])
            df.index.name = 'trial'
            df.to_csv(fileName, index=False)

    for key in design['base'].keys():
        if isinstance(design['base'][key][0],np.ndarray):
            design['base'][key]=design['base'][key][0].tolist()

        if len(design['base'][key]) != param['total_stimuli']:
            temp=[]
            try:
                temp=reshape3DToScalar(design['base'][key],temp) 
            except:
                temp=reshape2DToScalar(design['base'][key],temp) 
            design['base'][key]=temp

    fileName = subjID + '_base_design.csv'
    df=pd.DataFrame.from_dict(design['base'])
#     df.index.name = 'trial'
    df.to_csv(fileName, index=False)
    
    print('done')

# read back in from the csv form as the design variables
print('reading design data back in')

os.chdir(folder['design_output'])

design={}

design['base']={}
base_design_data=[]
filename = subjID+'_base_design.csv'
base_design_data = pd.read_csv(filename)
for key in list(base_design_data): 
    design['base'][key] = base_design_data[key].tolist()
    
design['block']=[[]]*param['num_blocks']
for b in range(param['num_blocks']):
    design_data=[]
    filename = subjID+'_session'+str(sessionNum)+'_block'+str(b)+'_design.csv'
    design_data = pd.read_csv(filename)

    design['block'][b]={}
    for key in list(design_data): 
        design['block'][b][key] = design_data[key].tolist()
    
print('done')



# encoding
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

if eeg == 1:
    #connect to eeg port and initialize data as 0 
    port = parallel.ParallelPort(address=0xD010)
    port.setData(0)

win = visual.Window(fullscr = winSize)
win.colorSpace = 'rgb'
win.color = [-1,-1,-1]
win.mouseVisible = False
win.flip()

loop_instruction = 1
while loop_instruction != 0:
    timer = core.Clock()
    output_encode = [[]]*param['num_blocks']
    #instruction
    instructions='In the first part of the experiment, a series of images will appear on the screen, one after the other.\n\nThese images will have a inverted-color square patch on them.\n\nWhen you see a inverted-color square patch on the left side of the picture, please press the button "J" as quickly as possible. If you see the patch on the right side of the picture, please press the button "K" as quickly as possible.\n\n\
Please make sure you always use the same fingers to press the buttons. \n\nYou will see many images, so we will give you breaks periodically.\n\nIf at any point of this part of the experiment that you want to quit, please press "Q".\n\nTo continue, please press "K".'
    message = visual.TextStim(win, text=instructions, height=0.05)
    message.draw()
    win.flip()
    keys=[]
    keys = event.waitKeys(keyList=['k','q'])
    if keys[0] == 'k':
        win.flip()
    if keys[0] == cancel_key:
        win.close()
        core.quit()

    # Sample stimuli
    os.chdir(folder['sample'])
    samplestim = glob.glob('*.jpg')

    instructions='This is an example of the image with a inverted-color square patch on the left side.\nYou should press "J" as soon as you find it.\nTo see another example, please press "J".'
    message = visual.TextStim(win, text=instructions, pos=(0.0,-0.25), height=0.05)
    image = visual.ImageStim(win, image=samplestim[0],pos=(0.0,param['img_y']/2.5),size=[param['img_x'],param['img_y']])
    grey_x=-0.1;grey_y=0.2;
    inverted_color_image = visual.ImageStim(win, image=samplestim[0],pos=(0-(grey_x),(param['img_y']/2.5)-(grey_y+param['img_y']/2.5)),size=[param['img_x'],param['img_y']])
    inverted_color_image.color*=-1
    temp_screenshot = visual.BufferImageStim(win, stim=[inverted_color_image], rect = param['inverted_patch_loc'] )
    screenshot = visual.ImageStim(win, image=temp_screenshot.image, 
                              pos=(grey_x,grey_y+param['img_y']/2.5),units='norm',size=param['patch_length'])
    message.draw()
    image.draw()
    screenshot.draw()
    win.flip()
    keys=[]
    keys = event.waitKeys(keyList=['j','q'])
    if keys[0] == 'j':
        win.flip()
    win.flip()
    if keys[0] == cancel_key:
        win.close()
        core.quit()
        
    instructions='This is an example of the image with a inverted-color square patch on the right side.\nYou should press "K" as soon as you find it.\nTo see another example, please press "K".'
    message = visual.TextStim(win, text=instructions, pos=(0.0,-0.25), height=0.05)
    image = visual.ImageStim(win, image=samplestim[1],pos=(0.0,param['img_y']/2.5),size=[param['img_x'],param['img_y']])
    grey_x=0.2;grey_y=0.3;
    inverted_color_image = visual.ImageStim(win, image=samplestim[1],pos=(0-grey_x,(param['img_y']/2.5)-(grey_y+param['img_y']/2.5)),size=[param['img_x'],param['img_y']])
    inverted_color_image.color*=-1
    temp_screenshot = visual.BufferImageStim(win, stim=[inverted_color_image], rect = param['inverted_patch_loc'] )
    screenshot = visual.ImageStim(win, image=temp_screenshot.image, 
                              pos=(grey_x,grey_y+param['img_y']/2.5),units='norm',size=param['patch_length'])
    message.draw()
    image.draw()
    screenshot.draw()
    win.flip()
    keys=[]
    keys = event.waitKeys(keyList=['k','q'])
    if keys[0] == 'k':
        win.flip()
    if keys[0] == cancel_key:
        win.close()
        core.quit()
        
    instructions='This is an example of the image with a inverted-color square patch on the left side.\nYou should press "J" as soon as you find it.\n\nNow please press "J".'
    message = visual.TextStim(win, text=instructions, pos=(0.0,-0.25), height=0.05)
    image = visual.ImageStim(win, image=samplestim[2],pos=(0.0,param['img_y']/2.5),size=[param['img_x'],param['img_y']])
    grey_x=-0.25;grey_y=-0.35;
    inverted_color_image = visual.ImageStim(win, image=samplestim[2],pos=(0-(grey_x),(param['img_y']/2.5)-(grey_y+param['img_y']/2.5)),size=[param['img_x'],param['img_y']])
    inverted_color_image.color*=-1
    temp_screenshot = visual.BufferImageStim(win, stim=[inverted_color_image], rect = param['inverted_patch_loc'] )
    screenshot = visual.ImageStim(win, image=temp_screenshot.image, 
                              pos=(grey_x,grey_y+param['img_y']/2.5),units='norm',size=param['patch_length'])
    message.draw()
    image.draw()
    screenshot.draw()
    win.flip()
    keys=[]
    keys = event.waitKeys(keyList=['j','q'])
    if keys[0] == 'j':
        win.flip()
    if keys[0] == cancel_key:
        win.close()
        core.quit()

    instructions='To start the experiment, please press "K".\n\nTo see the instructions again, please press "F".'
    message = visual.TextStim(win, text=instructions, pos=(0.0,0.0), height=0.05)
    message.draw()
    win.flip()
    keys=[]
    keys = event.waitKeys(keyList=['k','f','q'])
    if keys[0] == cancel_key:
        win.close()
        core.quit()
    if keys[0] != loop_instruction_key:
        loop_instruction = 0
    if keys[0] == 'k':
        win.flip()

            
for b in range(param['num_blocks']):
# for b in range(2):
    blockStart = timer.getTime()
    print('Start of the block:', b)  
    print('time now is:',datetime.datetime.now())

    # Load Stimuli
    block={}
    block['images'] = design['block'][b]['stimuli']
    block['condition_name'] = design['block'][b]['pairCondN']
    block['stimCond'] = design['block'][b]['stimCond']
    block['greypatch'] = design['block'][b]['greypatch']
    block['greypatch_positionx'] = design['block'][b]['greypatch_positionx']
    block['greypatch_positiony'] = design['block'][b]['greypatch_positiony']
    block['greypatch_positionLorR'] = design['block'][b]['greypatch_positionLorR']
    block['stimID'] = design['block'][b]['stimID']
    block['position'] = design['block'][b]['position']
    
    output_encode[b]={}
    output_encode[b]['key_pressed']=['']*param['trials_per_block']
    output_encode[b]['grey_patch'] = [0]*param['trials_per_block']
    output_encode[b]['response']=['']*param['trials_per_block']
    output_encode[b]['onset_abs'] = [0.0]*param['trials_per_block']
    output_encode[b]['onset_rel'] = [0.0]*param['trials_per_block']
    output_encode[b]['offset_abs'] = [0.0]*param['trials_per_block']
    output_encode[b]['offset_rel'] = [0.0]*param['trials_per_block']
    output_encode[b]['onset'] = [0.0]*param['trials_per_block']
    output_encode[b]['offset'] = [0.0]*param['trials_per_block']
    output_encode[b]['trial'] = [0.0]*param['trials_per_block']
    output_encode[b]['RT'] = [999.0]*param['trials_per_block']
    output_encode[b]['condition'] = ['']*param['trials_per_block']
    output_encode[b]['images'] = ['']*param['trials_per_block']
    output_encode[b]['stimID'] = [0.0]*param['trials_per_block']
    output_encode[b]['position'] = [0.0]*param['trials_per_block']
    output_encode[b]['block'] = [0.0]*param['trials_per_block']
    output_encode[b]['greyPatch_x'] = [0.0]*param['trials_per_block']
    output_encode[b]['greyPatch_y'] = [0.0]*param['trials_per_block']
    output_encode[b]['greyPatch_LorR'] = ['']*param['trials_per_block']
    
    mem_start = timer.getTime();
    for n in range(param['trials_per_block']):
#     for n in range(20):
        trial={}
        trial['start'] = timer.getTime()
        
        #draw fixation
        if eeg == 1:
            port.setData(param['fixationMarker'])
        while timer.getTime() < trial['start']+param['fixation_time']:
#             fixation = visual.Circle(win,radius=param['grey_radius'],pos=[0,0],fillColor=[1, 1, 1])
            # fixation cross
            fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
            fixation.draw()
            win.flip()
            
        print(n)
        output_encode[b]['onset'][n] = timer.getTime()
        output_encode[b]['onset_abs'][n] = timer.getTime() - blockStart
        output_encode[b]['onset_rel'][n] = timer.getTime() - mem_start
        output_encode[b]['trial'][n] = n
        output_encode[b]['condition'][n] = block['condition_name'][n]
        print('pairCond: ',output_encode[b]['condition'][n])
        
        trial['image'] = block['images'][n]
        trial['greypatch_positionLorR'] = block['greypatch_positionLorR'][n]
        output_encode[b]['images'][n]=trial['image']
        trial['condition_name'] = block['condition_name'][n] # get condition name
        trial['gray_patch']=0
        trial['response']=1
        
        keys=[]

        grey_x  = block['greypatch_positionx'][n]
        grey_y = block['greypatch_positiony'][n]
        grey_LorR = block['greypatch_positionLorR'][n]
        output_encode[b]['greyPatch_x'][n]=grey_x
        output_encode[b]['greyPatch_y'][n]=grey_y
        output_encode[b]['greyPatch_LorR'][n]=grey_LorR
#         print(grey_LorR)
        
        #draw image
        if eeg == 1:
            port.setData(block['stimID'][n]+(b*10))
            
        image = visual.ImageStim(win, image=trial['image'],size=[param['img_x'],param['img_y']],pos=(0,0),units='norm')
        
        if block['greypatch'][n] == 1:  
            output_encode[b]['grey_patch'][n] = 1
            
            inverted_color_image = visual.ImageStim(win, image=trial['image'],size=[param['img_x'],param['img_y']],pos=(0-grey_x,0-grey_y),units='norm')
            inverted_color_image.color*=-1
            temp_screenshot = visual.BufferImageStim(win, stim=[inverted_color_image], rect = param['inverted_patch_loc'] )
            screenshot = visual.ImageStim(win, image=temp_screenshot.image, 
                                      pos=(grey_x,grey_y),units='norm',size=param['patch_length'])
            
        image.draw() 
        screenshot.draw()
        
        fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
        fixation.draw()

        win.flip()
        print('stimID: ', block['stimID'][n]+(b*10))
        output_encode[b]['stimID'][n]=block['stimID'][n]+(b*10)
        output_encode[b]['position'][n]=block['position'][n]
                
        while timer.getTime() < trial['start']+param['stim_time']+param['fixation_time']:
            keys = event.getKeys()
            if keys != []:
                if keys[0] == cancel_key:
                    win.close()
                    core.quit()
                print(keys)
                output_encode[b]['key_pressed'][n]=keys[0]
                output_encode[b]['RT'][n] = timer.getTime() - trial['start']              
                  
                if output_encode[b]['key_pressed'][n] != '':
                    if grey_LorR == 'L':
                        if output_encode[b]['key_pressed'][n] == 'j':
                            #hit
                            trial['response'] = 1
                        elif output_encode[b]['key_pressed'][n] == 'k':
                            #wrong response
                            trial['response'] = -1
                    elif grey_LorR == 'R':
                        if output_encode[b]['key_pressed'][n] == 'k':
                            #hit
                            trial['response'] = 1
                        elif output_encode[b]['key_pressed'][n] == 'j':
                            #wrong response
                            trial['response'] = -1
                else:
                    #miss
                    trial['response'] = 0
            
                if trial['response'] == 1 and eeg == 1:
                    port.setData(param['correctResponseMarker'])
                    print(param['correctResponseMarker'])
                elif trial['response'] != 1 and eeg == 1:
                    port.setData(param['incorrectResponseMarker'])
                    print(param['incorrectResponseMarker'])
            
            
#         print('response:', trial['response'])
        output_encode[b]['response'][n]=trial['response']
            
        win.flip()
            
        output_encode[b]['offset'][n] = timer.getTime()
        output_encode[b]['offset_abs'][n] = timer.getTime() - blockStart
        output_encode[b]['offset_rel'][n] = timer.getTime() - mem_start
        output_encode[b]['block'][n] = b
    
    output_encode[b]['stimCondition']=block['stimCond']
            
    print('saving... encoding data')
    os.chdir(folder['output'])
    fileName = subjID + '_session' + str(sessionNum) + '_block' + str(b) + '_encoding_results.csv'
    df=pd.DataFrame.from_dict(output_encode[b])
    df.to_csv(fileName, index=False)
    
    #break
    if eeg == 1:
        port.setData(param['breakMarker']+(b*10))
        
    print('End of the block:', b)  
    print('time now is:',datetime.datetime.now())
    blockEnd=timer.getTime()
    blockDuration = blockEnd-blockStart
    print('block duration: ',blockDuration)
    
    if b != param['num_blocks']-1:
        print('End of the block:', b) 
#         countdown = 1
        countdown = param['breakTime']
        closing_instructions='This is the end of this set of images.\n\nPlease take a short break.\n\nWe will continue to show you images after the break ends.\n\nOnce again, please press "J" when you see the grey patch on the left side, press "K" when you see it on the right.\n\n00:15'
        message = visual.TextStim(win, text=closing_instructions, height=0.05)
        message.draw()
        core.wait(1)
        win.flip() 
        for count in reversed(range(0, countdown)):
            closing_instructions='This is the end of this set of images.\n\nPlease take a short break.\n\nWe will continue to show you images after the break ends.\n\nOnce again, please press "J" when you see the grey patch on the left side, press "K" when you see it on the right.\n\n00:'+str(count)
            message = visual.TextStim(win, text=closing_instructions, height=0.05)
            message.draw()
            core.wait(1)
            win.flip() 
        closing_instructions='This is the end of this set of images.\n\nPlease take a short break.\n\nWe will continue to show you images after the break ends.\n\nOnce again, please press "J" when you see the grey patch on the left side, press "K" when you see it on the right.\n\nTo start, please press "J".'
        message = visual.TextStim(win, text=closing_instructions, height=0.05)
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        if keys[0] == correct_key:
            win.flip()
        elif keys[0] == cancel_key:
            win.close()
            core.quit()
    else:
        closing_instructions='Congratulations, you have finished the first part of the experiment!\n\nNext, you will complete the second part.\n\nWe will give you instructions for this second part on the next page.\n\nTo continue, please press "J".'
        message = visual.TextStim(win, text=closing_instructions, height=0.05)
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        if keys[0] == correct_key:
            win.flip()
        elif keys[0] == cancel_key:
            win.close()
            core.quit()
               
win.close() 


#testing
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

if sessionNum == num_sessions-1:

    option_response = ['a','b']

    #read in stimuli name
    all_pairs_temp=[]
    all_pairs_temp = design['base']['stimuli']
    # group every two stim into a pair
    all_pairs_temp =[all_pairs_temp[i * 2:(i + 1) * 2] for i in range((len(all_pairs_temp) + 2 - 1) // 2 )]  
    randperm = np.random.permutation(param['total_pairs'])
    all_pairs=[all_pairs_temp[i] for i in randperm]
    #read in stimuli valence
    all_conditions_temp=[]
    all_conditions_temp = design['base']['pairCondN']
    all_conditions_temp =[all_conditions_temp[i * 2:(i + 1) * 2] for i in range((len(all_conditions_temp) + 2 - 1) // 2 )]  
    all_conditions=[all_conditions_temp[i] for i in randperm]
    #read in stimuli ID
    all_stimID_temp=[]
    all_stimID_temp = design['base']['stimID']
    all_stimID_temp =[all_stimID_temp[i * 2:(i + 1) * 2] for i in range((len(all_stimID_temp) + 2 - 1) // 2 )]  
    all_stimID=[all_stimID_temp[i] for i in randperm]
    
    output_testing_2AFC_endItem={}
    output_testing_2AFC_endItem['subjID'] = ['']*len(all_pairs)
    output_testing_2AFC_endItem['test'] = ['']*len(all_pairs)
    output_testing_2AFC_endItem['trial'] = ['']*len(all_pairs)
    output_testing_2AFC_endItem['condition'] = ['']*len(all_pairs)
    output_testing_2AFC_endItem['stimID'] = [0]*len(all_pairs)
    output_testing_2AFC_endItem['foil_stimID'] = [0]*len(all_pairs)
    output_testing_2AFC_endItem['correct_images'] = ['']*len(all_pairs)
    output_testing_2AFC_endItem['foil_images'] = ['']*len(all_pairs)
    output_testing_2AFC_endItem['response'] = [0]*len(all_pairs)
    output_testing_2AFC_endItem['choice_results']=['']*len(all_pairs)
    output_testing_2AFC_endItem['choiceHistory']=['']*len(all_pairs)
    output_testing_2AFC_endItem['decisionTime'] = [0]*len(all_pairs)
    output_testing_2AFC_endItem['option_order'] = [0]*len(all_pairs)
    output_testing_2AFC_endItem['either_end_item'] = ['']*len(all_pairs)
    output_testing_2AFC_endItem['foil_category'] = ['']*len(all_pairs)

    examplar_counter = {} #make sure one examplar gets foil of same category and one gets different category
    examplar_counter['rand_NeuNeu']=0;examplar_counter['rand_AffNeu']=0;examplar_counter['rand_AffAff']=0;
    examplar_counter['NeuNeu']=0;examplar_counter['NeuAff']=0;examplar_counter['AffNeu']=0;examplar_counter['AffAff']=0;
    foil_random_number_category = {}
    foil_random_number_category['rand_NeuNeu']=0;foil_random_number_category['rand_AffNeu']=0;foil_random_number_category['rand_AffAff']=0;
    foil_random_number_category['NeuNeu']=0;foil_random_number_category['NeuAff']=0;foil_random_number_category['AffNeu']=0;foil_random_number_category['AffAff']=0;
    category_indicator=[]
    category = ['same category','different category']
    foil_random_number_position = {}
    foil_random_number_position['rand_NeuNeu']=0;foil_random_number_position['rand_AffNeu']=0;foil_random_number_position['rand_AffAff']=0;
    foil_random_number_position['NeuNeu']=0;foil_random_number_position['NeuAff']=0;foil_random_number_position['AffNeu']=0;foil_random_number_position['AffAff']=0;
    position_indicator=[]
    position = ['first item foil','second item foil']
    foil_pairs=['']*len(all_pairs)
    correct_pairs=[]

    for n in range(len(all_pairs)):

        output_testing_2AFC_endItem['condition'][n] = all_conditions[n]
        output_testing_2AFC_endItem['stimID'][n] = all_stimID[n]
        output_testing_2AFC_endItem['foil_stimID'][n] = all_stimID[n][:]
        correct_pairs.append([all_pairs[n][0],all_pairs[n][1]])

        if examplar_counter[all_conditions[n][0]] == 0:
            if 'rand' in all_conditions[n][0]:
                foil_random_number_position[all_conditions[n][0]] = np.random.permutation(param['num_examplars_unstructured']+1)
                foil_random_number_category[all_conditions[n][0]] = np.random.permutation(param['num_examplars_unstructured']+1)
            else:
                foil_random_number_position[all_conditions[n][0]] = np.random.permutation(param['num_examplars_structured'])
                foil_random_number_category[all_conditions[n][0]] = np.random.permutation(param['num_examplars_structured'])

            position_indicator.append(foil_random_number_position[all_conditions[n][0]][0]%2)
            output_testing_2AFC_endItem['either_end_item'][n]=position[position_indicator[n]%2]
            
            category_indicator.append(foil_random_number_category[all_conditions[n][0]][0]%2)
#             output_testing_2AFC_endItem['foil_category'][n]=category[category_indicator[n]%2]
            #here we only use different category foil
            output_testing_2AFC_endItem['foil_category'][n]='different category'
            
        else:
            if 'rand' in all_conditions[n][0]:
                position_indicator.append(foil_random_number_position[all_conditions[n][0]][examplar_counter[all_conditions[n][0]]]%2)
                category_indicator.append(foil_random_number_category[all_conditions[n][0]][examplar_counter[all_conditions[n][0]]]%2)
            else:
                position_indicator.append(foil_random_number_position[all_conditions[n][0]][examplar_counter[all_conditions[n][0]]]%2)
                category_indicator.append(foil_random_number_category[all_conditions[n][0]][examplar_counter[all_conditions[n][0]]]%2)

            output_testing_2AFC_endItem['either_end_item'][n]=position[position_indicator[n]%2]
#             output_testing_2AFC_endItem['foil_category'][n]=category[category_indicator[n]%2]
            #here we only use different category foil
            output_testing_2AFC_endItem['foil_category'][n]='different category'
        
        if output_testing_2AFC_endItem['either_end_item'][n]=='first item foil':
            if 'negative' in all_pairs[n][0]:
                if output_testing_2AFC_endItem['foil_category'][n]=='same category':
                    Aff_stim = all_pairs[n][0]
                    while Aff_stim == all_pairs[n][0] or Aff_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neg']))))
                        Aff_stim = stimuli['neg'][random_number]
                    del stimuli['neg'][random_number]
                    foil_pairs[n]=[Aff_stim,all_pairs[n][1]]
                    output_testing_2AFC_endItem['foil_stimID'][n][0]=param['negFoilImageMarker']
                
                if output_testing_2AFC_endItem['foil_category'][n]=='different category':
                    Neu_stim = all_pairs[n][0]
                    while Neu_stim == all_pairs[n][0] or Neu_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neu']))))
                        Neu_stim = stimuli['neu'][random_number]
                    del stimuli['neu'][random_number]
                    foil_pairs[n]=[Neu_stim,all_pairs[n][1]]
                    output_testing_2AFC_endItem['foil_stimID'][n][0]=param['neuFoilImageMarker']
            
            elif 'neutral' in all_pairs[n][0]:
                if output_testing_2AFC_endItem['foil_category'][n]=='same category':
                    Neu_stim = all_pairs[n][0]
                    while Neu_stim == all_pairs[n][0] or Neu_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neu']))))
                        Neu_stim = stimuli['neu'][random_number]
                    del stimuli['neu'][random_number]
                    foil_pairs[n]=[Neu_stim,all_pairs[n][1]]
                    output_testing_2AFC_endItem['foil_stimID'][n][0]=param['neuFoilImageMarker']
                
                if output_testing_2AFC_endItem['foil_category'][n]=='different category':
                    Aff_stim = all_pairs[n][0]
                    while Aff_stim == all_pairs[n][0] or Aff_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neg']))))
                        Aff_stim = stimuli['neg'][random_number]
                    del stimuli['neg'][random_number]
                    foil_pairs[n]=[Aff_stim,all_pairs[n][1]]
                    output_testing_2AFC_endItem['foil_stimID'][n][0]=param['negFoilImageMarker']
                                     
        elif output_testing_2AFC_endItem['either_end_item'][n]=='second item foil':
            if 'negative' in all_pairs[n][1]:
                if output_testing_2AFC_endItem['foil_category'][n]=='same category':
                    Aff_stim = all_pairs[n][1]
                    while Aff_stim == all_pairs[n][0] or Aff_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neg']))))
                        Aff_stim = stimuli['neg'][random_number]
                    del stimuli['neg'][random_number]
                    foil_pairs[n]=[all_pairs[n][0],Aff_stim]
                    output_testing_2AFC_endItem['foil_stimID'][n][1]=param['negFoilImageMarker']
                    
                if output_testing_2AFC_endItem['foil_category'][n]=='different category':
                    Neu_stim = all_pairs[n][1]
                    while Neu_stim == all_pairs[n][0] or Neu_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neu']))))
                        Neu_stim = stimuli['neu'][random_number]
                    del stimuli['neu'][random_number]
                    foil_pairs[n]=[all_pairs[n][0],Neu_stim]
                    output_testing_2AFC_endItem['foil_stimID'][n][1]=param['neuFoilImageMarker']
                    
            elif 'neutral' in all_pairs[n][1]:
                if output_testing_2AFC_endItem['foil_category'][n]=='same category':
                    Neu_stim = all_pairs[n][1]
                    while Neu_stim == all_pairs[n][0] or Neu_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neu']))))
                        Neu_stim = stimuli['neu'][random_number]
                    del stimuli['neu'][random_number]
                    foil_pairs[n]=[all_pairs[n][0],Neu_stim]
                    output_testing_2AFC_endItem['foil_stimID'][n][1]=param['neuFoilImageMarker']
                
                if output_testing_2AFC_endItem['foil_category'][n]=='different category':
                    Aff_stim = all_pairs[n][1]
                    while Aff_stim == all_pairs[n][0] or Aff_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neg']))))
                        Aff_stim = stimuli['neg'][random_number]
                    del stimuli['neg'][random_number]
                    foil_pairs[n]=[all_pairs[n][0],Aff_stim]
                    output_testing_2AFC_endItem['foil_stimID'][n][1]=param['negFoilImageMarker']
                
        examplar_counter[all_conditions[n][0]]=examplar_counter[all_conditions[n][0]]+1
        

    win = visual.Window(fullscr = winSize)
    win.colorSpace = 'rgb'
    win.color = [-1,-1,-1]
    win.flip()
    timer = core.Clock()
    win.mouseVisible = False

    loop_instruction = 1
    while loop_instruction != 0:
        intro1='In this part of the experiment, you will see two groups of images: Group A and Group B. Each group includes a few images, which will appear one after the other.\n\nYour job is to indicate which group looks most FAMILIAR. Please pay close attention when the images are shown, as they will not be repeated.\n\nFirst, we will show you Groups A and B. After seeing all groups, you will decide which group looks more familiar.\n\nTo start the experiment, please press "J".\nTo see the instructions again, please press "F".'
        message = visual.TextStim(win, text=intro1, pos=(0.0, 0.0), height=0.05)
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        win.flip()
        if keys[0] == cancel_key:
            win.close()
            core.quit()
        if keys[0] != loop_instruction_key:
            loop_instruction = 0
        win.flip()
        
    for n in range(len(correct_pairs)):
        output_testing_2AFC_endItem['trial'][n] = n
        correct_option = correct_pairs[n]
        foil_option = foil_pairs[n]
        output_testing_2AFC_endItem['correct_images'][n] = correct_option
        output_testing_2AFC_endItem['foil_images'][n] = foil_option
        
        #randomize option order, 0 is where the correct option is
        option_order = np.random.permutation(2)
        print('option order ',option_order)
        print('stim marker ',all_stimID[n])
        
        #this loop the two options
        count = 0
        for m in range(2):
            if count == 0:
                intro='You are going to see Group A. Some images will appear, one after the other.'
                message = visual.TextStim(win, text=intro, height=0.05)
                message.draw()
                win.flip()
                core.wait(2.5)
                win.flip()

            elif count == 1:
                intro='You are going to see Group B. Some images will appear, one after the other.'
                message = visual.TextStim(win, text=intro, height=0.05)
                message.draw()
                win.flip() 
                core.wait(2.5)
                win.flip()
        
            if option_order[m] == 0:
                print(option_order[m],correct_option)
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                if eeg == 1:
                    port.setData(param['fixationMarker'])
                win.flip()
                core.wait(0.5)
                
                image = visual.ImageStim(win, image=correct_option[0],size=[param['img_x'],param['img_y']],pos=(0,0))
                if eeg == 1:
                    port.setData(all_stimID[n][0])
                image.draw() 
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                win.flip()
                core.wait(1.5)
                win.flip()
                
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                if eeg == 1:
                    port.setData(param['fixationMarker'])
                win.flip()
                core.wait(0.5)
                
                image = visual.ImageStim(win, image=correct_option[1],size=[param['img_x'],param['img_y']],pos=(0,0))
                if eeg == 1:
                    port.setData(all_stimID[n][1])
                image.draw() 
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                win.flip()
                core.wait(1.5)
                win.flip()
                
                keys = event.getKeys()
                if keys != []:
                    if keys[0] == cancel_key:
                        win.close()
                        core.quit()
                
            elif option_order[m] == 1:
                print(option_order[m],foil_option)
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                if eeg == 1:
                    port.setData(param['fixationMarker'])
                win.flip()
                core.wait(0.5)
                
                image = visual.ImageStim(win, image=foil_option[0],size=[param['img_x'],param['img_y']],pos=(0,0))
                if eeg == 1 and output_testing_2AFC_endItem['either_end_item'] =='first item foil':
                    port.setData(foil_stimID)
                elif eeg == 1 and output_testing_2AFC_endItem['either_end_item'] =='first item foil':
                    port.setData(all_stimID[n][0])
                image.draw() 
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                win.flip()
                core.wait(1.5)
                win.flip()
                
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                if eeg == 1:
                    port.setData(param['fixationMarker'])
                win.flip()
                core.wait(0.5)
                
                image = visual.ImageStim(win, image=foil_option[1],size=[param['img_x'],param['img_y']],pos=(0,0))
                if eeg == 1 and output_testing_2AFC_endItem['either_end_item'] =='second item foil':
                    port.setData(foil_stimID)
                elif eeg == 1 and output_testing_2AFC_endItem['either_end_item'] =='second item foil':
                    port.setData(all_stimID[n][1])
                image.draw() 
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                win.flip()
                core.wait(1.5)
                win.flip()
                
                keys = event.getKeys()
                if keys != []:
                    if keys[0] == cancel_key:
                        win.close()
                        core.quit()
                
            count = count+1

        intro='If the image sequence in Option A looked more familiar, choose "A".\n\nIf the image sequence in Option B looked more familiar, choose "B".\n\nPlease use the left and right arrow key to indicate your choice of option.\n\nTo continue, press "Enter".'
        message = visual.TextStim(win, text=intro, pos=(0.0, 0.4), height=0.05)
        ratingScale = visual.RatingScale(win,pos=(0.0,-0.4), noMouse=True, markerColor='black', textColor="black", lineColor='black',showAccept=False, acceptKeys='return', choices=['A', 'B'], markerStart=0.5)
        rectangle = visual.Rect(win, width=1.2, height=0.4, opacity=0.55,pos=(0.0,-0.4), lineColor='white', lineColorSpace='rgb', fillColor='white')

        while ratingScale.noResponse:
            message.draw()
            rectangle.draw()
            ratingScale.draw()
            win.flip()
            
        output_testing_2AFC_endItem['choice_results'][n] = ratingScale.getRating()
        output_testing_2AFC_endItem['choiceHistory'][n] = ratingScale.getHistory()
        output_testing_2AFC_endItem['decisionTime'][n] = ratingScale.getRT()
        
        win.flip()
        
        #option_order 0 is where the right option is 
        #so 2 is the right option
        if ratingScale.getRating() == 'A' and option_order[0] == 0:
            output_testing_2AFC_endItem['response'][n] = 2
        elif ratingScale.getRating() == 'B' and option_order[1] == 0:
            output_testing_2AFC_endItem['response'][n] = 2

        if ratingScale.getRating() == 'A' and option_order[0] == 1 and output_testing_2AFC_endItem['foil_category'][n] == 'same category':
            output_testing_2AFC_endItem['response'][n] = 1
        elif ratingScale.getRating() == 'B' and option_order[1] == 1 and output_testing_2AFC_endItem['foil_category'][n] == 'same category':
            output_testing_2AFC_endItem['response'][n] = 1
            
        print(output_testing_2AFC_endItem['response'])

        output_testing_2AFC_endItem['option_order'][n] = option_order
        output_testing_2AFC_endItem['subjID'][n] = subjID
        output_testing_2AFC_endItem['test'][n] = '2AFC_endItem_test'

    intro1='Thank you for finishing this part of the experiment.\n\nTo continue, press "J".'
    message = visual.TextStim(win, text=intro1, pos=(0.0, 0.0), height=0.05)
    message.draw()
    win.flip()
    keys = event.waitKeys(keyList=ok_response)
    win.flip()
    if keys[0] == cancel_key:
        win.close()
        core.quit()

    # In[5]:

    # save file
    print('saving... testing 2AFC data')
    os.chdir(folder['output'])
    fileName = subjID +  '_testing_2AFC_endItem_results.csv'
    print(fileName)
    df=pd.DataFrame.from_dict(output_testing_2AFC_endItem)
    df.to_csv(fileName, index=False)

    win.close()


#rating
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
if sessionNum == num_sessions-1:

    valence_scale = folder['scale']+'valence.png'
    arousal_scale = folder['scale']+'arousal.png'

    all_image = design['base']['stimuli']
    randperm = np.random.permutation(param['total_stimuli'])
    all_image_randomized = [all_image[i] for i in randperm]
    all_stimCondition = design['base']['stimCond']
    all_stimCondition_randomized = [all_stimCondition[i] for i in randperm]    
    all_stimID = design['base']['stimID']
    all_stimID_randomized = [all_stimID[i] for i in randperm] 

    win = visual.Window(fullscr = winSize)
    win.colorSpace = 'rgb'
    win.color = [-1,-1,-1]
    win.mouseVisible = False
    win.flip()

    loop_instruction = 1
    while loop_instruction != 0:
        intro1='For this part of the experiment, we will show you the images you saw earlier.\n\nFor each image, you will rate how it made you feel.\n\nNote that you will not be able to quit the experiment during this part of the experiment. If you do not wish to continue, please press “Q” to quit now.\n\nTo continue, please press "J".'
        message = visual.TextStim(win, text=intro1,pos=(0.0, 0.0), height=0.05)
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        win.flip()
        if keys[0] == cancel_key:
            win.close()
            core.quit()
        intro2='First, you will rate whether the image made you feel pleasant or unpleasant. In other words, we want you to rate your emotional feelings: whether you felt positive/pleasant/good, or negative/unpleasant/bad.\n\nSecond, you will rate whether the image made you feel activated or deactivated. In other words, we want you to rate the physical feelings in your body: whether you felt activated/excited/worked-up, or deactivated/tired/bored.\n\nWe will show you both of the scales and provide some more examples.\n\nTo continue, please press "J".'
        message = visual.TextStim(win, text=intro2, pos=(0.0, 0.0), height=0.05)
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        win.flip()
        if keys[0] == cancel_key:
            win.close()
            core.quit()
        intro3='For each image, you will use the icons below to answer how pleasant or unpleasant the image made you feel. This is a rating of your emotional feelings.\n\nIf the image made you feel negative or unpleasant, then you should select icons on the left side of the scale.\n\nIf the image made you feel positive or pleasant, then you should select icons on the right side of the scale.\n\nIf the image made you feel very little or no emotion, then you should select icons around the middle of the scale.\n\nTo continue, please press "J".'
        message = visual.TextStim(win, text=intro3, pos=(0.0, 0.2), height=0.05)
        val_scale = visual.ImageStim(win, image=valence_scale,pos=(0.0,-0.6),size=(1.2,0.2), contrast = -1)
        val_scale.draw()
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        win.flip()
        if keys[0] == cancel_key:
            win.close()
            core.quit()
        intro4='For each image, you will also rate how activated or deactivated the image made you feel. This is a rating of the physical feelings in your body.\n\nThink of when you feel most deactivated, like right before you fall asleep, or when you feel calm, depressed, or bored. If the image made you feel this way, then you should select icons on the left side of the scale.\n\nThink of when you feel most activated, like when you have run up a flight of stairs, had many cups of coffee, or feel excited, angry, or afraid. If the image made you feel this way, then you should select icons on the right side of the scale.\n\nThink of when you have felt awake but not too worked up, like when you are doing something engaging but not too activating, e.g. chores around the house. If the image made you feel this way, then you should select icons around the middle of the scale.\n\nTo continue, please press "J".'
        message = visual.TextStim(win, text=intro4, pos=(0.0, 0.2), height=0.05)
        arou_scale = visual.ImageStim(win, image=arousal_scale,pos=(0.0,-0.6), size=(1.2,0.2), contrast = -1)
        arou_scale.draw()
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        win.flip()
        if keys[0] == cancel_key:
            win.close()
            core.quit()
        intro5='We will show you images in a moment. For each, you will rate your emotional feeling (unpleasant–pleasant) and the physical feelings in your body (deactivated–activated).\n\nYou will rate the scales one at a time beneath the image. Please select an icon using the arrow keys for the scales.\n\nAfter selecting both ratings, press "Enter" or "Return" to continue to the next trial.\n\nPlease note that you will not be able to quit during this part of the experiment. If you do not wish to continue, please press "Q" to quit now.\n\nTo start the experiment, please press "J".\nTo see the instructions again, please press "F".'
        message = visual.TextStim(win, text=intro5, pos=(0.0, 0.0), height=0.05)
        win.flip() 
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        win.flip()
        if keys[0] == cancel_key:
            win.close()
            core.quit()
        if keys[0] != loop_instruction_key:
            loop_instruction = 0
        win.flip()
        
    output_rating={}
    output_rating['trial'] = ['']*len(all_image_randomized)
    output_rating['image'] = ['']*len(all_image_randomized)
    output_rating['val_rating'] = [0.0]*len(all_image_randomized)
    output_rating['val_decisionTime'] = [0.0]*len(all_image_randomized)
    output_rating['val_choiceHistory'] = ['']*len(all_image_randomized)
    output_rating['arou_rating'] = [0.0]*len(all_image_randomized)
    output_rating['arou_decisionTime'] = [0.0]*len(all_image_randomized)
    output_rating['stimCondition'] = ['']*len(all_image_randomized)
    output_rating['stimID'] = [0.0]*len(all_image_randomized)
    output_rating['arou_choiceHistory'] = ['']*len(all_image_randomized)

    for n in range(len(all_image_randomized)):
#     for n in range(2):
        output_rating['trial'][n] = n
        output_rating['image'][n] = all_image_randomized[n]
        output_rating['stimCondition'][n] = all_stimCondition_randomized[n]
        output_rating['stimID'][n] = all_stimID_randomized[n]

        image = visual.ImageStim(win, image=all_image_randomized[n],pos=(0.0,param['img_y']/2.5),size=[param['img_x'],param['img_y']])
        val_labels = 'Extremely Negative','Extremely Positive'
        val_scale = visual.ImageStim(win, image=valence_scale,pos=(0.0,-0.35),size=(1.0,0.16), contrast = -1)
        val_scale_grey = visual.ImageStim(win, image=valence_scale,pos=(0.0,-0.35),size=(1.0,0.16), contrast = -0.5)
        val_ratingScale = visual.RatingScale(win,pos=(0.0,-0.45), low=1, high=9,scale=None, labels = val_labels, markerColor='white', textColor="white",stretch=1.5, acceptKeys='return', lineColor='white',showAccept=False, noMouse=True, markerStart=5)
        arou_labels = 'Extremely Deactivated','Extremely Activated'
        arou_scale = visual.ImageStim(win, image=arousal_scale,pos=(0.0,-0.65),size=(1.0,0.16), contrast = -1)
        arou_scale_grey = visual.ImageStim(win, image=arousal_scale,pos=(0.0,-0.65),size=(1.0,0.16), contrast = -0.5)
        arou_ratingScale = visual.RatingScale(win,pos=(0.0,-0.75), labels = arou_labels,scale=None,low=1, high=9, markerColor='white', textColor="white",stretch=1.5, acceptKeys='return', lineColor='white',showAccept=False, noMouse=True, markerStart=5)

        if eeg == 1:
            port.setData(all_stimID_randomized[n])
        while val_ratingScale.noResponse:
            image.draw()
            val_scale.draw()
            val_ratingScale.draw()
#             arou_scale_grey.draw()
            win.flip()
        output_rating['val_rating'][n] = val_ratingScale.getRating()
        output_rating['val_decisionTime'][n] = val_ratingScale.getRT()
        output_rating['val_choiceHistory'][n] = val_ratingScale.getHistory()
        while arou_ratingScale.noResponse:
            image.draw()
            arou_scale.draw()
            arou_ratingScale.draw() 
#             val_scale_grey.draw()
            win.flip()
        output_rating['arou_rating'][n] = arou_ratingScale.getRating()
        output_rating['arou_decisionTime'][n] = arou_ratingScale.getRT()
        output_rating['arou_choiceHistory'][n] = arou_ratingScale.getHistory()

    intro='Congratulations, you have finished this part of the experiment!\n\nTo continue, please press "J".'
    message = visual.TextStim(win, text=intro, pos=(0.0, 0.2), height=0.05)
    win.flip() 
    message.draw()
    win.flip() 
    keys = event.waitKeys(keyList=ok_response)
    win.flip()
    if keys[0] == cancel_key:
        win.close()
        core.quit()

    # save block-wise file
    print('saving... rating data')
    os.chdir(folder['output'])
    fileName = subjID + '_rating_results.csv'
    print(fileName)
    df=pd.DataFrame.from_dict(output_rating)
    df.to_csv(fileName, index=False)

    win.close()





# encoding
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

if eeg == 1:
    #connect to eeg port and initialize data as 0 
    port = parallel.ParallelPort(address=D010)
    port.setData(0)

win = visual.Window(fullscr = winSize)
win.colorSpace = 'rgb'
win.color = [-1,-1,-1]
win.mouseVisible = False
win.flip()

loop_instruction = 1
while loop_instruction != 0:
    timer = core.Clock()
    output_encode = [[]]*param['num_blocks']
    #instruction
    instructions='In the first part of the experiment, a series of images will appear on the screen, one after the other.\n\nThese images will have a inverted-color square patch on them.\n\nWhen you see a inverted-color square patch on the left side of the picture, please press the button "J" as quickly as possible. If you see the patch on the right side of the picture, please press the button "K" as quickly as possible.\n\n\
Please make sure you always use the same fingers to press the buttons. \n\nYou will see many images, so we will give you breaks periodically.\n\nIf at any point of this part of the experiment that you want to quit, please press "Q".\n\nTo continue, please press "K".'
    message = visual.TextStim(win, text=instructions, height=0.05)
    message.draw()
    win.flip()
    keys=[]
    keys = event.waitKeys(keyList=['k','q'])
    if keys[0] == 'k':
        win.flip()
    if keys[0] == cancel_key:
        win.close()
        core.quit()

    # Sample stimuli
    os.chdir(folder['sample'])
    samplestim = glob.glob('*.jpg')

    instructions='This is an example of the image with a inverted-color square patch on the left side.\nYou should press "J" as soon as you find it.\nTo see another example, please press "J".'
    message = visual.TextStim(win, text=instructions, pos=(0.0,-0.25), height=0.05)
    image = visual.ImageStim(win, image=samplestim[0],pos=(0.0,param['img_y']/2.5),size=[param['img_x'],param['img_y']])
    grey_x=-0.1;grey_y=0.2;
    inverted_color_image = visual.ImageStim(win, image=samplestim[0],pos=(0-(grey_x),(param['img_y']/2.5)-(grey_y+param['img_y']/2.5)),size=[param['img_x'],param['img_y']])
    inverted_color_image.color*=-1
    temp_screenshot = visual.BufferImageStim(win, stim=[inverted_color_image], rect = param['inverted_patch_loc'] )
    screenshot = visual.ImageStim(win, image=temp_screenshot.image, 
                              pos=(grey_x,grey_y+param['img_y']/2.5),units='norm',size=param['patch_length'])
    message.draw()
    image.draw()
    screenshot.draw()
    win.flip()
    keys=[]
    keys = event.waitKeys(keyList=['j','q'])
    if keys[0] == 'j':
        win.flip()
    win.flip()
    if keys[0] == cancel_key:
        win.close()
        core.quit()
        
    instructions='This is an example of the image with a inverted-color square patch on the right side.\nYou should press "K" as soon as you find it.\nTo see another example, please press "K".'
    message = visual.TextStim(win, text=instructions, pos=(0.0,-0.25), height=0.05)
    image = visual.ImageStim(win, image=samplestim[1],pos=(0.0,param['img_y']/2.5),size=[param['img_x'],param['img_y']])
    grey_x=0.2;grey_y=0.3;
    inverted_color_image = visual.ImageStim(win, image=samplestim[1],pos=(0-grey_x,(param['img_y']/2.5)-(grey_y+param['img_y']/2.5)),size=[param['img_x'],param['img_y']])
    inverted_color_image.color*=-1
    temp_screenshot = visual.BufferImageStim(win, stim=[inverted_color_image], rect = param['inverted_patch_loc'] )
    screenshot = visual.ImageStim(win, image=temp_screenshot.image, 
                              pos=(grey_x,grey_y+param['img_y']/2.5),units='norm',size=param['patch_length'])
    message.draw()
    image.draw()
    screenshot.draw()
    win.flip()
    keys=[]
    keys = event.waitKeys(keyList=['k','q'])
    if keys[0] == 'k':
        win.flip()
    if keys[0] == cancel_key:
        win.close()
        core.quit()
        
    instructions='This is an example of the image with a inverted-color square patch on the left side.\nYou should press "J" as soon as you find it.\n\nNow please press "J".'
    message = visual.TextStim(win, text=instructions, pos=(0.0,-0.25), height=0.05)
    image = visual.ImageStim(win, image=samplestim[2],pos=(0.0,param['img_y']/2.5),size=[param['img_x'],param['img_y']])
    grey_x=-0.25;grey_y=-0.35;
    inverted_color_image = visual.ImageStim(win, image=samplestim[2],pos=(0-(grey_x),(param['img_y']/2.5)-(grey_y+param['img_y']/2.5)),size=[param['img_x'],param['img_y']])
    inverted_color_image.color*=-1
    temp_screenshot = visual.BufferImageStim(win, stim=[inverted_color_image], rect = param['inverted_patch_loc'] )
    screenshot = visual.ImageStim(win, image=temp_screenshot.image, 
                              pos=(grey_x,grey_y+param['img_y']/2.5),units='norm',size=param['patch_length'])
    message.draw()
    image.draw()
    screenshot.draw()
    win.flip()
    keys=[]
    keys = event.waitKeys(keyList=['j','q'])
    if keys[0] == 'j':
        win.flip()
    if keys[0] == cancel_key:
        win.close()
        core.quit()

    instructions='To start the experiment, please press "K".\n\nTo see the instructions again, please press "F".'
    message = visual.TextStim(win, text=instructions, pos=(0.0,0.0), height=0.05)
    message.draw()
    win.flip()
    keys=[]
    keys = event.waitKeys(keyList=['k','f','q'])
    if keys[0] == cancel_key:
        win.close()
        core.quit()
    if keys[0] != loop_instruction_key:
        loop_instruction = 0
    if keys[0] == 'k':
        win.flip()

            
for b in range(param['num_blocks']):
# for b in range(2):
    blockStart = timer.getTime()
    print('Start of the block:', b)  
    print('time now is:',datetime.datetime.now())

    # Load Stimuli
    block={}
    block['images'] = design['block'][b]['stimuli']
    block['condition_name'] = design['block'][b]['pairCondN']
    block['stimCond'] = design['block'][b]['stimCond']
    block['greypatch'] = design['block'][b]['greypatch']
    block['greypatch_positionx'] = design['block'][b]['greypatch_positionx']
    block['greypatch_positiony'] = design['block'][b]['greypatch_positiony']
    block['greypatch_positionLorR'] = design['block'][b]['greypatch_positionLorR']
    block['stimID'] = design['block'][b]['stimID']
    block['position'] = design['block'][b]['position']
    
    output_encode[b]={}
    output_encode[b]['key_pressed']=['']*param['trials_per_block']
    output_encode[b]['grey_patch'] = [0]*param['trials_per_block']
    output_encode[b]['response']=['']*param['trials_per_block']
    output_encode[b]['onset_abs'] = [0.0]*param['trials_per_block']
    output_encode[b]['onset_rel'] = [0.0]*param['trials_per_block']
    output_encode[b]['offset_abs'] = [0.0]*param['trials_per_block']
    output_encode[b]['offset_rel'] = [0.0]*param['trials_per_block']
    output_encode[b]['onset'] = [0.0]*param['trials_per_block']
    output_encode[b]['offset'] = [0.0]*param['trials_per_block']
    output_encode[b]['trial'] = [0.0]*param['trials_per_block']
    output_encode[b]['RT'] = [999.0]*param['trials_per_block']
    output_encode[b]['condition'] = ['']*param['trials_per_block']
    output_encode[b]['images'] = ['']*param['trials_per_block']
    output_encode[b]['stimID'] = [0.0]*param['trials_per_block']
    output_encode[b]['position'] = [0.0]*param['trials_per_block']
    output_encode[b]['block'] = [0.0]*param['trials_per_block']
    output_encode[b]['greyPatch_x'] = [0.0]*param['trials_per_block']
    output_encode[b]['greyPatch_y'] = [0.0]*param['trials_per_block']
    output_encode[b]['greyPatch_LorR'] = ['']*param['trials_per_block']
    
    mem_start = timer.getTime();
    for n in range(param['trials_per_block']):
#     for n in range(20):
        trial={}
        trial['start'] = timer.getTime()
        
        #draw fixation
        if eeg == 1:
            port.setData(param['fixationMarker'])
        while timer.getTime() < trial['start']+param['fixation_time']:
#             fixation = visual.Circle(win,radius=param['grey_radius'],pos=[0,0],fillColor=[1, 1, 1])
            # fixation cross
            fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
            fixation.draw()
            win.flip()
            
        print(n)
        output_encode[b]['onset'][n] = timer.getTime()
        output_encode[b]['onset_abs'][n] = timer.getTime() - blockStart
        output_encode[b]['onset_rel'][n] = timer.getTime() - mem_start
        output_encode[b]['trial'][n] = n
        output_encode[b]['condition'][n] = block['condition_name'][n]
        print('pairCond: ',output_encode[b]['condition'][n])
        
        trial['image'] = block['images'][n]
        trial['greypatch_positionLorR'] = block['greypatch_positionLorR'][n]
        output_encode[b]['images'][n]=trial['image']
        trial['condition_name'] = block['condition_name'][n] # get condition name
        trial['gray_patch']=0
        trial['response']=1
        
        keys=[]

        grey_x  = block['greypatch_positionx'][n]
        grey_y = block['greypatch_positiony'][n]
        grey_LorR = block['greypatch_positionLorR'][n]
        output_encode[b]['greyPatch_x'][n]=grey_x
        output_encode[b]['greyPatch_y'][n]=grey_y
        output_encode[b]['greyPatch_LorR'][n]=grey_LorR
#         print(grey_LorR)
        
        #draw image
        if eeg == 1:
            port.setData(block['stimID'][n]+(b*10))
            
        image = visual.ImageStim(win, image=trial['image'],size=[param['img_x'],param['img_y']],pos=(0,0),units='norm')
        
        if block['greypatch'][n] == 1:  
            output_encode[b]['grey_patch'][n] = 1
            
            inverted_color_image = visual.ImageStim(win, image=trial['image'],size=[param['img_x'],param['img_y']],pos=(0-grey_x,0-grey_y),units='norm')
            inverted_color_image.color*=-1
            temp_screenshot = visual.BufferImageStim(win, stim=[inverted_color_image], rect = param['inverted_patch_loc'] )
            screenshot = visual.ImageStim(win, image=temp_screenshot.image, 
                                      pos=(grey_x,grey_y),units='norm',size=param['patch_length'])
            
        image.draw() 
        screenshot.draw()
        
        fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
        fixation.draw()

        win.flip()
        print('stimID: ', block['stimID'][n]+(b*10))
        output_encode[b]['stimID'][n]=block['stimID'][n]+(b*10)
        output_encode[b]['position'][n]=block['position'][n]
                
        while timer.getTime() < trial['start']+param['stim_time']+param['fixation_time']:
            keys = event.getKeys()
            if keys != []:
                if keys[0] == cancel_key:
                    win.close()
                    core.quit()
                print(keys)
                output_encode[b]['key_pressed'][n]=keys[0]
                output_encode[b]['RT'][n] = timer.getTime() - trial['start']
                  
        if output_encode[b]['key_pressed'][n] != '':
            if grey_LorR == 'L':
                if output_encode[b]['key_pressed'][n] == 'j':
                    #hit
                    trial['response'] = 1
                elif output_encode[b]['key_pressed'][n] == 'k':
                    #wrong response
                    trial['response'] = -1
            elif grey_LorR == 'R':
                if output_encode[b]['key_pressed'][n] == 'k':
                    #hit
                    trial['response'] = 1
                elif output_encode[b]['key_pressed'][n] == 'j':
                    #wrong response
                    trial['response'] = -1
        else:
            #miss
            trial['response'] = 0
            
        if trial['response'] == 1 and eeg == 1:
            port.setData(param['correctResponseMarker'])
        elif trial['response'] != 1 and eeg == 1:
            port.setData(param['incorrectResponseMarker'])
            
#         print('response:', trial['response'])
        output_encode[b]['response'][n]=trial['response']
            
        win.flip()
            
        output_encode[b]['offset'][n] = timer.getTime()
        output_encode[b]['offset_abs'][n] = timer.getTime() - blockStart
        output_encode[b]['offset_rel'][n] = timer.getTime() - mem_start
        output_encode[b]['block'][n] = b
    
    output_encode[b]['stimCondition']=block['stimCond']
            
    print('saving... encoding data')
    os.chdir(folder['output'])
    fileName = subjID + '_session' + str(sessionNum) + '_block' + str(b) + '_encoding_results.csv'
    df=pd.DataFrame.from_dict(output_encode[b])
    df.to_csv(fileName, index=False)
    
    #break
    if eeg == 1:
        port.setData(param['breakMarker']+(b*10))
        
    print('End of the block:', b)  
    print('time now is:',datetime.datetime.now())
    blockEnd=timer.getTime()
    blockDuration = blockEnd-blockStart
    print('block duration: ',blockDuration)
    
    if b != param['num_blocks']-1:
        print('End of the block:', b) 
#         countdown = 1
        countdown = param['breakTime']
        closing_instructions='This is the end of this set of images.\n\nPlease take a short break.\n\nWe will continue to show you images after the break ends.\n\nOnce again, please press "J" when you see the grey patch on the left side, press "K" when you see it on the right.\n\n00:15'
        message = visual.TextStim(win, text=closing_instructions, height=0.05)
        message.draw()
        core.wait(1)
        win.flip() 
        for count in reversed(range(0, countdown)):
            closing_instructions='This is the end of this set of images.\n\nPlease take a short break.\n\nWe will continue to show you images after the break ends.\n\nOnce again, please press "J" when you see the grey patch on the left side, press "K" when you see it on the right.\n\n00:'+str(count)
            message = visual.TextStim(win, text=closing_instructions, height=0.05)
            message.draw()
            core.wait(1)
            win.flip() 
        closing_instructions='This is the end of this set of images.\n\nPlease take a short break.\n\nWe will continue to show you images after the break ends.\n\nOnce again, please press "J" when you see the grey patch on the left side, press "K" when you see it on the right.\n\nTo start, please press "J".'
        message = visual.TextStim(win, text=closing_instructions, height=0.05)
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        if keys[0] == correct_key:
            win.flip()
        elif keys[0] == cancel_key:
            win.close()
            core.quit()
    else:
        closing_instructions='Congratulations, you have finished the first part of the experiment!\n\nNext, you will complete the second part.\n\nWe will give you instructions for this second part on the next page.\n\nTo continue, please press "J".'
        message = visual.TextStim(win, text=closing_instructions, height=0.05)
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        if keys[0] == correct_key:
            win.flip()
        elif keys[0] == cancel_key:
            win.close()
            core.quit()
               
win.close() 


#testing
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

if sessionNum == num_sessions-1:

    option_response = ['a','b']

    #read in stimuli name
    all_pairs_temp=[]
    all_pairs_temp = design['base']['stimuli']
    # group every two stim into a pair
    all_pairs_temp =[all_pairs_temp[i * 2:(i + 1) * 2] for i in range((len(all_pairs_temp) + 2 - 1) // 2 )]  
    randperm = np.random.permutation(param['total_pairs'])
    all_pairs=[all_pairs_temp[i] for i in randperm]
    #read in stimuli valence
    all_conditions_temp=[]
    all_conditions_temp = design['base']['pairCondN']
    all_conditions_temp =[all_conditions_temp[i * 2:(i + 1) * 2] for i in range((len(all_conditions_temp) + 2 - 1) // 2 )]  
    all_conditions=[all_conditions_temp[i] for i in randperm]
    #read in stimuli ID
    all_stimID_temp=[]
    all_stimID_temp = design['base']['stimID']
    all_stimID_temp =[all_stimID_temp[i * 2:(i + 1) * 2] for i in range((len(all_stimID_temp) + 2 - 1) // 2 )]  
    all_stimID=[all_stimID_temp[i] for i in randperm]
    
    output_testing_2AFC_endItem={}
    output_testing_2AFC_endItem['subjID'] = ['']*len(all_pairs)
    output_testing_2AFC_endItem['test'] = ['']*len(all_pairs)
    output_testing_2AFC_endItem['trial'] = ['']*len(all_pairs)
    output_testing_2AFC_endItem['condition'] = ['']*len(all_pairs)
    output_testing_2AFC_endItem['stimID'] = [0]*len(all_pairs)
    output_testing_2AFC_endItem['foil_stimID'] = [0]*len(all_pairs)
    output_testing_2AFC_endItem['correct_images'] = ['']*len(all_pairs)
    output_testing_2AFC_endItem['foil_images'] = ['']*len(all_pairs)
    output_testing_2AFC_endItem['response'] = [0]*len(all_pairs)
    output_testing_2AFC_endItem['choice_results']=['']*len(all_pairs)
    output_testing_2AFC_endItem['choiceHistory']=['']*len(all_pairs)
    output_testing_2AFC_endItem['decisionTime'] = [0]*len(all_pairs)
    output_testing_2AFC_endItem['option_order'] = [0]*len(all_pairs)
    output_testing_2AFC_endItem['either_end_item'] = ['']*len(all_pairs)
    output_testing_2AFC_endItem['foil_category'] = ['']*len(all_pairs)

    examplar_counter = {} #make sure one examplar gets foil of same category and one gets different category
    examplar_counter['rand_NeuNeu']=0;examplar_counter['rand_AffNeu']=0;examplar_counter['rand_AffAff']=0;
    examplar_counter['NeuNeu']=0;examplar_counter['NeuAff']=0;examplar_counter['AffNeu']=0;examplar_counter['AffAff']=0;
    foil_random_number_category = {}
    foil_random_number_category['rand_NeuNeu']=0;foil_random_number_category['rand_AffNeu']=0;foil_random_number_category['rand_AffAff']=0;
    foil_random_number_category['NeuNeu']=0;foil_random_number_category['NeuAff']=0;foil_random_number_category['AffNeu']=0;foil_random_number_category['AffAff']=0;
    category_indicator=[]
    category = ['same category','different category']
    foil_random_number_position = {}
    foil_random_number_position['rand_NeuNeu']=0;foil_random_number_position['rand_AffNeu']=0;foil_random_number_position['rand_AffAff']=0;
    foil_random_number_position['NeuNeu']=0;foil_random_number_position['NeuAff']=0;foil_random_number_position['AffNeu']=0;foil_random_number_position['AffAff']=0;
    position_indicator=[]
    position = ['first item foil','second item foil']
    foil_pairs=['']*len(all_pairs)
    correct_pairs=[]

    for n in range(len(all_pairs)):

        output_testing_2AFC_endItem['condition'][n] = all_conditions[n]
        output_testing_2AFC_endItem['stimID'][n] = all_stimID[n]
        output_testing_2AFC_endItem['foil_stimID'][n] = all_stimID[n][:]
        correct_pairs.append([all_pairs[n][0],all_pairs[n][1]])

        if examplar_counter[all_conditions[n][0]] == 0:
            if 'rand' in all_conditions[n][0]:
                foil_random_number_position[all_conditions[n][0]] = np.random.permutation(param['num_examplars_unstructured']+1)
                foil_random_number_category[all_conditions[n][0]] = np.random.permutation(param['num_examplars_unstructured']+1)
            else:
                foil_random_number_position[all_conditions[n][0]] = np.random.permutation(param['num_examplars_structured'])
                foil_random_number_category[all_conditions[n][0]] = np.random.permutation(param['num_examplars_structured'])

            position_indicator.append(foil_random_number_position[all_conditions[n][0]][0]%2)
            output_testing_2AFC_endItem['either_end_item'][n]=position[position_indicator[n]%2]
            
            category_indicator.append(foil_random_number_category[all_conditions[n][0]][0]%2)
#             output_testing_2AFC_endItem['foil_category'][n]=category[category_indicator[n]%2]
            #here we only use different category foil
            output_testing_2AFC_endItem['foil_category'][n]='different category'
            
        else:
            if 'rand' in all_conditions[n][0]:
                position_indicator.append(foil_random_number_position[all_conditions[n][0]][examplar_counter[all_conditions[n][0]]]%2)
                category_indicator.append(foil_random_number_category[all_conditions[n][0]][examplar_counter[all_conditions[n][0]]]%2)
            else:
                position_indicator.append(foil_random_number_position[all_conditions[n][0]][examplar_counter[all_conditions[n][0]]]%2)
                category_indicator.append(foil_random_number_category[all_conditions[n][0]][examplar_counter[all_conditions[n][0]]]%2)

            output_testing_2AFC_endItem['either_end_item'][n]=position[position_indicator[n]%2]
#             output_testing_2AFC_endItem['foil_category'][n]=category[category_indicator[n]%2]
            #here we only use different category foil
            output_testing_2AFC_endItem['foil_category'][n]='different category'
        
        if output_testing_2AFC_endItem['either_end_item'][n]=='first item foil':
            if 'negative' in all_pairs[n][0]:
                if output_testing_2AFC_endItem['foil_category'][n]=='same category':
                    Aff_stim = all_pairs[n][0]
                    while Aff_stim == all_pairs[n][0] or Aff_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neg']))))
                        Aff_stim = stimuli['neg'][random_number]
                    del stimuli['neg'][random_number]
                    foil_pairs[n]=[Aff_stim,all_pairs[n][1]]
                    output_testing_2AFC_endItem['foil_stimID'][n][0]=param['negFoilImageMarker']
                
                if output_testing_2AFC_endItem['foil_category'][n]=='different category':
                    Neu_stim = all_pairs[n][0]
                    while Neu_stim == all_pairs[n][0] or Neu_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neu']))))
                        Neu_stim = stimuli['neu'][random_number]
                    del stimuli['neu'][random_number]
                    foil_pairs[n]=[Neu_stim,all_pairs[n][1]]
                    output_testing_2AFC_endItem['foil_stimID'][n][0]=param['neuFoilImageMarker']
            
            elif 'neutral' in all_pairs[n][0]:
                if output_testing_2AFC_endItem['foil_category'][n]=='same category':
                    Neu_stim = all_pairs[n][0]
                    while Neu_stim == all_pairs[n][0] or Neu_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neu']))))
                        Neu_stim = stimuli['neu'][random_number]
                    del stimuli['neu'][random_number]
                    foil_pairs[n]=[Neu_stim,all_pairs[n][1]]
                    output_testing_2AFC_endItem['foil_stimID'][n][0]=param['neuFoilImageMarker']
                
                if output_testing_2AFC_endItem['foil_category'][n]=='different category':
                    Aff_stim = all_pairs[n][0]
                    while Aff_stim == all_pairs[n][0] or Aff_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neg']))))
                        Aff_stim = stimuli['neg'][random_number]
                    del stimuli['neg'][random_number]
                    foil_pairs[n]=[Aff_stim,all_pairs[n][1]]
                    output_testing_2AFC_endItem['foil_stimID'][n][0]=param['negFoilImageMarker']
                                     
        elif output_testing_2AFC_endItem['either_end_item'][n]=='second item foil':
            if 'negative' in all_pairs[n][1]:
                if output_testing_2AFC_endItem['foil_category'][n]=='same category':
                    Aff_stim = all_pairs[n][1]
                    while Aff_stim == all_pairs[n][0] or Aff_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neg']))))
                        Aff_stim = stimuli['neg'][random_number]
                    del stimuli['neg'][random_number]
                    foil_pairs[n]=[all_pairs[n][0],Aff_stim]
                    output_testing_2AFC_endItem['foil_stimID'][n][1]=param['negFoilImageMarker']
                    
                if output_testing_2AFC_endItem['foil_category'][n]=='different category':
                    Neu_stim = all_pairs[n][1]
                    while Neu_stim == all_pairs[n][0] or Neu_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neu']))))
                        Neu_stim = stimuli['neu'][random_number]
                    del stimuli['neu'][random_number]
                    foil_pairs[n]=[all_pairs[n][0],Neu_stim]
                    output_testing_2AFC_endItem['foil_stimID'][n][1]=param['neuFoilImageMarker']
                    
            elif 'neutral' in all_pairs[n][1]:
                if output_testing_2AFC_endItem['foil_category'][n]=='same category':
                    Neu_stim = all_pairs[n][1]
                    while Neu_stim == all_pairs[n][0] or Neu_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neu']))))
                        Neu_stim = stimuli['neu'][random_number]
                    del stimuli['neu'][random_number]
                    foil_pairs[n]=[all_pairs[n][0],Neu_stim]
                    output_testing_2AFC_endItem['foil_stimID'][n][1]=param['neuFoilImageMarker']
                
                if output_testing_2AFC_endItem['foil_category'][n]=='different category':
                    Aff_stim = all_pairs[n][1]
                    while Aff_stim == all_pairs[n][0] or Aff_stim == all_pairs[n][1]:
                        random_number = random.choice(list(range(len(stimuli['neg']))))
                        Aff_stim = stimuli['neg'][random_number]
                    del stimuli['neg'][random_number]
                    foil_pairs[n]=[all_pairs[n][0],Aff_stim]
                    output_testing_2AFC_endItem['foil_stimID'][n][1]=param['negFoilImageMarker']
                
        examplar_counter[all_conditions[n][0]]=examplar_counter[all_conditions[n][0]]+1
        

    win = visual.Window(fullscr = winSize)
    win.colorSpace = 'rgb'
    win.color = [-1,-1,-1]
    win.flip()
    timer = core.Clock()
    win.mouseVisible = False

    loop_instruction = 1
    while loop_instruction != 0:
        intro1='In this part of the experiment, you will see two groups of images: Group A and Group B. Each group includes a few images, which will appear one after the other.\n\nYour job is to indicate which group looks most FAMILIAR. Please pay close attention when the images are shown, as they will not be repeated.\n\nFirst, we will show you Groups A and B. After seeing all groups, you will decide which group looks more familiar.\n\nTo start the experiment, please press "J".\nTo see the instructions again, please press "F".'
        message = visual.TextStim(win, text=intro1, pos=(0.0, 0.0), height=0.05)
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        win.flip()
        if keys[0] == cancel_key:
            win.close()
            core.quit()
        if keys[0] != loop_instruction_key:
            loop_instruction = 0
        win.flip()
        
    for n in range(len(correct_pairs)):
        output_testing_2AFC_endItem['trial'][n] = n
        correct_option = correct_pairs[n]
        foil_option = foil_pairs[n]
        output_testing_2AFC_endItem['correct_images'][n] = correct_option
        output_testing_2AFC_endItem['foil_images'][n] = foil_option
        
        #randomize option order, 0 is where the correct option is
        option_order = np.random.permutation(2)
        print('option order ',option_order)
        print('stim marker ',all_stimID[n])
        
        #this loop the two options
        count = 0
        for m in range(2):
            if count == 0:
                intro='You are going to see Group A. Some images will appear, one after the other.'
                message = visual.TextStim(win, text=intro, height=0.05)
                message.draw()
                win.flip()
                core.wait(2.5)
                win.flip()

            elif count == 1:
                intro='You are going to see Group B. Some images will appear, one after the other.'
                message = visual.TextStim(win, text=intro, height=0.05)
                message.draw()
                win.flip() 
                core.wait(2.5)
                win.flip()
        
            if option_order[m] == 0:
                print(option_order[m],correct_option)
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                if eeg == 1:
                    port.setData(param['fixationMarker'])
                win.flip()
                core.wait(0.5)
                
                image = visual.ImageStim(win, image=correct_option[0],size=[param['img_x'],param['img_y']],pos=(0,0))
                if eeg == 1:
                    port.setData(all_stimID[n][0])
                image.draw() 
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                win.flip()
                core.wait(1.5)
                win.flip()
                
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                if eeg == 1:
                    port.setData(param['fixationMarker'])
                win.flip()
                core.wait(0.5)
                
                image = visual.ImageStim(win, image=correct_option[1],size=[param['img_x'],param['img_y']],pos=(0,0))
                if eeg == 1:
                    port.setData(all_stimID[n][1])
                image.draw() 
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                win.flip()
                core.wait(1.5)
                win.flip()
                
                keys = event.getKeys()
                if keys != []:
                    if keys[0] == cancel_key:
                        win.close()
                        core.quit()
                
            elif option_order[m] == 1:
                print(option_order[m],foil_option)
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                if eeg == 1:
                    port.setData(param['fixationMarker'])
                win.flip()
                core.wait(0.5)
                
                image = visual.ImageStim(win, image=foil_option[0],size=[param['img_x'],param['img_y']],pos=(0,0))
                if eeg == 1 and output_testing_2AFC_endItem['either_end_item'] =='first item foil':
                    port.setData(foil_stimID)
                elif eeg == 1 and output_testing_2AFC_endItem['either_end_item'] =='first item foil':
                    port.setData(all_stimID[n][0])
                image.draw() 
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                win.flip()
                core.wait(1.5)
                win.flip()
                
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                if eeg == 1:
                    port.setData(param['fixationMarker'])
                win.flip()
                core.wait(0.5)
                
                image = visual.ImageStim(win, image=foil_option[1],size=[param['img_x'],param['img_y']],pos=(0,0))
                if eeg == 1 and output_testing_2AFC_endItem['either_end_item'] =='second item foil':
                    port.setData(foil_stimID)
                elif eeg == 1 and output_testing_2AFC_endItem['either_end_item'] =='second item foil':
                    port.setData(all_stimID[n][1])
                image.draw() 
                fixation = visual.ShapeStim(win, vertices=param['fixation_location'],units='pix',lineWidth=param['fixation_width'],closeShape=False,lineColor="white")
                fixation.draw()
                win.flip()
                core.wait(1.5)
                win.flip()
                
                keys = event.getKeys()
                if keys != []:
                    if keys[0] == cancel_key:
                        win.close()
                        core.quit()
                
            count = count+1

        intro='If the image sequence in Option A looked more familiar, choose "A".\n\nIf the image sequence in Option B looked more familiar, choose "B".\n\nPlease use the left and right arrow key to indicate your choice of option.\n\nTo continue, press "Enter".'
        message = visual.TextStim(win, text=intro, pos=(0.0, 0.4), height=0.05)
        ratingScale = visual.RatingScale(win,pos=(0.0,-0.4), noMouse=True, markerColor='black', textColor="black", lineColor='black',showAccept=False, acceptKeys='return', choices=['A', 'B'], markerStart=0.5)
        rectangle = visual.Rect(win, width=1.2, height=0.4, opacity=0.55,pos=(0.0,-0.4), lineColor='white', lineColorSpace='rgb', fillColor='white')

        while ratingScale.noResponse:
            message.draw()
            rectangle.draw()
            ratingScale.draw()
            win.flip()
            
        output_testing_2AFC_endItem['choice_results'][n] = ratingScale.getRating()
        output_testing_2AFC_endItem['choiceHistory'][n] = ratingScale.getHistory()
        output_testing_2AFC_endItem['decisionTime'][n] = ratingScale.getRT()
        
        win.flip()
        
        #option_order 0 is where the right option is 
        #so 2 is the right option
        if ratingScale.getRating() == 'A' and option_order[0] == 0:
            output_testing_2AFC_endItem['response'][n] = 2
        elif ratingScale.getRating() == 'B' and option_order[1] == 0:
            output_testing_2AFC_endItem['response'][n] = 2

        if ratingScale.getRating() == 'A' and option_order[0] == 1 and output_testing_2AFC_endItem['foil_category'][n] == 'same category':
            output_testing_2AFC_endItem['response'][n] = 1
        elif ratingScale.getRating() == 'B' and option_order[1] == 1 and output_testing_2AFC_endItem['foil_category'][n] == 'same category':
            output_testing_2AFC_endItem['response'][n] = 1
            
        print(output_testing_2AFC_endItem['response'])

        output_testing_2AFC_endItem['option_order'][n] = option_order
        output_testing_2AFC_endItem['subjID'][n] = subjID
        output_testing_2AFC_endItem['test'][n] = '2AFC_endItem_test'

    intro1='Thank you for finishing this part of the experiment.\n\nTo continue, press "J".'
    message = visual.TextStim(win, text=intro1, pos=(0.0, 0.0), height=0.05)
    message.draw()
    win.flip()
    keys = event.waitKeys(keyList=ok_response)
    win.flip()
    if keys[0] == cancel_key:
        win.close()
        core.quit()

    # In[5]:

    # save file
    print('saving... testing 2AFC data')
    os.chdir(folder['output'])
    fileName = subjID +  '_testing_2AFC_endItem_results.csv'
    print(fileName)
    df=pd.DataFrame.from_dict(output_testing_2AFC_endItem)
    df.to_csv(fileName, index=False)

    win.close()


#rating
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
if sessionNum == num_sessions-1:

    valence_scale = folder['scale']+'valence.png'
    arousal_scale = folder['scale']+'arousal.png'

    all_image = design['base']['stimuli']
    randperm = np.random.permutation(param['total_stimuli'])
    all_image_randomized = [all_image[i] for i in randperm]
    all_stimCondition = design['base']['stimCond']
    all_stimCondition_randomized = [all_stimCondition[i] for i in randperm]    
    all_stimID = design['base']['stimID']
    all_stimID_randomized = [all_stimID[i] for i in randperm] 

    win = visual.Window(fullscr = winSize)
    win.colorSpace = 'rgb'
    win.color = [-1,-1,-1]
    win.mouseVisible = False
    win.flip()

    loop_instruction = 1
    while loop_instruction != 0:
        intro1='For this part of the experiment, we will show you the images you saw earlier.\n\nFor each image, you will rate how it made you feel.\n\nNote that you will not be able to quit the experiment during this part of the experiment. If you do not wish to continue, please press “Q” to quit now.\n\nTo continue, please press "J".'
        message = visual.TextStim(win, text=intro1,pos=(0.0, 0.0), height=0.05)
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        win.flip()
        if keys[0] == cancel_key:
            win.close()
            core.quit()
        intro2='First, you will rate whether the image made you feel pleasant or unpleasant. In other words, we want you to rate your emotional feelings: whether you felt positive/pleasant/good, or negative/unpleasant/bad.\n\nSecond, you will rate whether the image made you feel activated or deactivated. In other words, we want you to rate the physical feelings in your body: whether you felt activated/excited/worked-up, or deactivated/tired/bored.\n\nWe will show you both of the scales and provide some more examples.\n\nTo continue, please press "J".'
        message = visual.TextStim(win, text=intro2, pos=(0.0, 0.0), height=0.05)
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        win.flip()
        if keys[0] == cancel_key:
            win.close()
            core.quit()
        intro3='For each image, you will use the icons below to answer how pleasant or unpleasant the image made you feel. This is a rating of your emotional feelings.\n\nIf the image made you feel negative or unpleasant, then you should select icons on the left side of the scale.\n\nIf the image made you feel positive or pleasant, then you should select icons on the right side of the scale.\n\nIf the image made you feel very little or no emotion, then you should select icons around the middle of the scale.\n\nTo continue, please press "J".'
        message = visual.TextStim(win, text=intro3, pos=(0.0, 0.2), height=0.05)
        val_scale = visual.ImageStim(win, image=valence_scale,pos=(0.0,-0.6),size=(1.2,0.2), contrast = -1)
        val_scale.draw()
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        win.flip()
        if keys[0] == cancel_key:
            win.close()
            core.quit()
        intro4='For each image, you will also rate how activated or deactivated the image made you feel. This is a rating of the physical feelings in your body.\n\nThink of when you feel most deactivated, like right before you fall asleep, or when you feel calm, depressed, or bored. If the image made you feel this way, then you should select icons on the left side of the scale.\n\nThink of when you feel most activated, like when you have run up a flight of stairs, had many cups of coffee, or feel excited, angry, or afraid. If the image made you feel this way, then you should select icons on the right side of the scale.\n\nThink of when you have felt awake but not too worked up, like when you are doing something engaging but not too activating, e.g. chores around the house. If the image made you feel this way, then you should select icons around the middle of the scale.\n\nTo continue, please press "J".'
        message = visual.TextStim(win, text=intro4, pos=(0.0, 0.2), height=0.05)
        arou_scale = visual.ImageStim(win, image=arousal_scale,pos=(0.0,-0.6), size=(1.2,0.2), contrast = -1)
        arou_scale.draw()
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        win.flip()
        if keys[0] == cancel_key:
            win.close()
            core.quit()
        intro5='We will show you images in a moment. For each, you will rate your emotional feeling (unpleasant–pleasant) and the physical feelings in your body (deactivated–activated).\n\nYou will rate the scales one at a time beneath the image. Please select an icon using the arrow keys for the scales.\n\nAfter selecting both ratings, press "Enter" or "Return" to continue to the next trial.\n\nPlease note that you will not be able to quit during this part of the experiment. If you do not wish to continue, please press "Q" to quit now.\n\nTo start the experiment, please press "J".\nTo see the instructions again, please press "F".'
        message = visual.TextStim(win, text=intro5, pos=(0.0, 0.0), height=0.05)
        win.flip() 
        message.draw()
        win.flip() 
        keys = event.waitKeys(keyList=ok_response)
        win.flip()
        if keys[0] == cancel_key:
            win.close()
            core.quit()
        if keys[0] != loop_instruction_key:
            loop_instruction = 0
        win.flip()
        
    output_rating={}
    output_rating['trial'] = ['']*len(all_image_randomized)
    output_rating['image'] = ['']*len(all_image_randomized)
    output_rating['val_rating'] = [0.0]*len(all_image_randomized)
    output_rating['val_decisionTime'] = [0.0]*len(all_image_randomized)
    output_rating['val_choiceHistory'] = ['']*len(all_image_randomized)
    output_rating['arou_rating'] = [0.0]*len(all_image_randomized)
    output_rating['arou_decisionTime'] = [0.0]*len(all_image_randomized)
    output_rating['stimCondition'] = ['']*len(all_image_randomized)
    output_rating['stimID'] = [0.0]*len(all_image_randomized)
    output_rating['arou_choiceHistory'] = ['']*len(all_image_randomized)

    for n in range(len(all_image_randomized)):
#     for n in range(2):
        output_rating['trial'][n] = n
        output_rating['image'][n] = all_image_randomized[n]
        output_rating['stimCondition'][n] = all_stimCondition_randomized[n]
        output_rating['stimID'][n] = all_stimID_randomized[n]

        image = visual.ImageStim(win, image=all_image_randomized[n],pos=(0.0,param['img_y']/2.5),size=[param['img_x'],param['img_y']])
        val_labels = 'Extremely Negative','Extremely Positive'
        val_scale = visual.ImageStim(win, image=valence_scale,pos=(0.0,-0.35),size=(1.0,0.16), contrast = -1)
        val_scale_grey = visual.ImageStim(win, image=valence_scale,pos=(0.0,-0.35),size=(1.0,0.16), contrast = -0.5)
        val_ratingScale = visual.RatingScale(win,pos=(0.0,-0.45), low=1, high=9,scale=None, labels = val_labels, markerColor='white', textColor="white",stretch=1.5, acceptKeys='return', lineColor='white',showAccept=False, noMouse=True, markerStart=5)
        arou_labels = 'Extremely Deactivated','Extremely Activated'
        arou_scale = visual.ImageStim(win, image=arousal_scale,pos=(0.0,-0.65),size=(1.0,0.16), contrast = -1)
        arou_scale_grey = visual.ImageStim(win, image=arousal_scale,pos=(0.0,-0.65),size=(1.0,0.16), contrast = -0.5)
        arou_ratingScale = visual.RatingScale(win,pos=(0.0,-0.75), labels = arou_labels,scale=None,low=1, high=9, markerColor='white', textColor="white",stretch=1.5, acceptKeys='return', lineColor='white',showAccept=False, noMouse=True, markerStart=5)

        if eeg == 1:
            port.setData(all_stimID_randomized[n])
        while val_ratingScale.noResponse:
            image.draw()
            val_scale.draw()
            val_ratingScale.draw()
#             arou_scale_grey.draw()
            win.flip()
        output_rating['val_rating'][n] = val_ratingScale.getRating()
        output_rating['val_decisionTime'][n] = val_ratingScale.getRT()
        output_rating['val_choiceHistory'][n] = val_ratingScale.getHistory()
        while arou_ratingScale.noResponse:
            image.draw()
            arou_scale.draw()
            arou_ratingScale.draw() 
#             val_scale_grey.draw()
            win.flip()
        output_rating['arou_rating'][n] = arou_ratingScale.getRating()
        output_rating['arou_decisionTime'][n] = arou_ratingScale.getRT()
        output_rating['arou_choiceHistory'][n] = arou_ratingScale.getHistory()

    intro='Congratulations, you have finished this part of the experiment!\n\nTo continue, please press "J".'
    message = visual.TextStim(win, text=intro, pos=(0.0, 0.2), height=0.05)
    win.flip() 
    message.draw()
    win.flip() 
    keys = event.waitKeys(keyList=ok_response)
    win.flip()
    if keys[0] == cancel_key:
        win.close()
        core.quit()

    # save block-wise file
    print('saving... rating data')
    os.chdir(folder['output'])
    fileName = subjID + '_rating_results.csv'
    print(fileName)
    df=pd.DataFrame.from_dict(output_rating)
    df.to_csv(fileName, index=False)

    win.close()



