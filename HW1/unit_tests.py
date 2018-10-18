import random

import matplotlib.pyplot as plt
import numpy as np

import ID3
import parse


def testID3AndEvaluate():
  data = [dict(a=1, b=0, Class=1), dict(a=1, b=1, Class=1)]
  tree = ID3.ID3(data, 0)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=1, b=0))
    if ans != 1:
      print("ID3 test failed.")
    else:
      print("ID3 test succeeded.")
  else:
    print("ID3 test failed -- no tree returned")

def testPruning():
  # data = [dict(a=1, b=1, c=1, Class=0), dict(a=1, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1), dict(a=0, b=0, c=0, Class=1), dict(a=0, b=0, c=1, Class=0)]
  # validationData = [dict(a=0, b=0, c=1, Class=1)]
  data = [dict(a=0, b=1, c=1, d=0, Class=1), dict(a=0, b=0, c=1, d=0, Class=0), dict(a=0, b=1, c=0, d=0, Class=1), dict(a=1, b=0, c=1, d=0, Class=0), dict(a=1, b=1, c=0, d=0, Class=0), dict(a=1, b=1, c=0, d=1, Class=0), dict(a=1, b=1, c=1, d=0, Class=0)]
  validationData = [dict(a=0, b=0, c=1, d=0, Class=1), dict(a=1, b=1, c=1, d=1, Class = 0)]
  tree = ID3.ID3(data, 0)
  ID3.prune(tree, validationData)
  if tree != None:
    ans = ID3.evaluate(tree, dict(a=0, b=0, c=1, d=0))
    if ans != 1:
      print("pruning test failed.")
    else:
      print("pruning test succeeded.")
  else:
    print("pruning test failed -- no tree returned.")


def testID3AndTest():
  trainData = [dict(a=1, b=0, c=0, Class=1), dict(a=1, b=1, c=0, Class=1), 
  dict(a=0, b=0, c=0, Class=0), dict(a=0, b=1, c=0, Class=1)]
  testData = [dict(a=1, b=0, c=1, Class=1), dict(a=1, b=1, c=1, Class=1), 
  dict(a=0, b=0, c=1, Class=0), dict(a=0, b=1, c=1, Class=0)]
  tree = ID3.ID3(trainData, 0)
  fails = 0
  if tree != None:
    acc = ID3.test(tree, trainData)
    if acc == 1.0:
      print("testing on train data succeeded.")
    else:
      print("testing on train data failed.")
      fails = fails + 1
    acc = ID3.test(tree, testData)
    if acc == 0.75:
      print("testing on test data succeeded.")
    else:
      print("testing on test data failed.")
      fails = fails + 1
    if fails > 0:
      print("Failures: ", fails)
    else:
      print("testID3AndTest succeeded.")
  else:
    print("testID3andTest failed -- no tree returned.")	

# inFile - string location of the house data file
def testPruningOnHouseData(inFile):
  data = parse.parse(inFile)
  with_prunings = []
  without_prunings = []
  for size in range(10, 301):
    withPruning = []
    withoutPruning = []
    for i in range(100):
      random.shuffle(data)
      train_prune = data[:7 * size // 10]
      valid_prune = data[7 * size // 10:size]
      test = data[size:]
      train_without_prune = data[:size]
      # valid = data[size:(size+size//5)]
      # test = data[(size+size//5):(size+2*size//5)]

      tree = ID3.ID3(train_prune, 'democrat')
      '''
      acc = ID3.test(tree, train)
      # print("training accuracy: ",acc)
      acc = ID3.test(tree, valid)
      # print("validation accuracy: ",acc)
      acc = ID3.test(tree, test)
      # print("test accuracy: ",acc)
      '''

      ID3.prune(tree, valid_prune)
      '''
      acc = ID3.test(tree, train)
      # print("pruned tree train accuracy: ",acc)
      acc = ID3.test(tree, valid)
      # print("pruned tree validation accuracy: ",acc)
      '''
      acc = ID3.test(tree, test)
      # print("pruned tree test accuracy: ",acc)
      withPruning.append(acc)
      # tree = ID3.ID3(train + valid, 'democrat')
      '''The training data should be only train when pruning is not performed.'''
      tree = ID3.ID3(train_without_prune, 'democrat')
      acc = ID3.test(tree, test)
      # print("no pruning test accuracy: ",acc)
      withoutPruning.append(acc)
    # print(withPruning)
    # print(withoutPruning)
    with_prunings.append(sum(withPruning) / len(withPruning))
    without_prunings.append(sum(withoutPruning) / len(withoutPruning))
    # print("average with pruning",sum(withPruning)/len(withPruning),
    # " without: ",sum(withoutPruning)/len(withoutPruning))
  print("with_prunnings")
  print(with_prunings)
  print("without_prunnings")
  print(without_prunings)
  x_axis = np.linspace(10, 300, 291)
  fig = plt.figure(figsize=(7, 3))
  plt.plot(x_axis, with_prunings)
  plt.plot(x_axis, without_prunings)
  plt.legend(['With Pruning', 'Without Pruning'], loc='best', bbox_to_anchor=(1, 0.5))
  plt.show()

  '''
  x = pd.DataFrame(with_prunings)
  x.to_csv('x2.csv')
  y = pd.DataFrame(without_prunings)
  y.to_csv('y2.csv')
'''

# testID3AndEvaluate()
testPruningOnHouseData("house_votes_84.data")
