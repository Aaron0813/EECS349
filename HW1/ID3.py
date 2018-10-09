from node import Node
import math


def calc_ent(examples, target):
    element_nums = len(examples)
    feature_count = {}
    # calc the the number of unique elements
    for example in examples:
        current_attribute = example[target]
        if current_attribute not in feature_count.keys():
            feature_count[current_attribute] = 0
        feature_count[current_attribute] += 1
    ent = 0.0
    for attribute in feature_count:
        prob = float(feature_count[attribute]) / element_nums
        ent -= prob * math.log(prob, 2)
    return ent


# return subset whose label is not equal to value
def split_data(examples, label, value):
    subset = []
    for example in examples:
        if example[label] == value:
            subset.append(example)
    return subset


# find the most entropy and return the feature name---为什么有的时候bestFeature会是-1
def choose_best_feature(examples, target):
    attribute_nums = len(examples[0]) - 1
    # Entropy
    baseEntropy = calc_ent(examples, target)
    best_info_gain = 0
    best_feature = -1
    current_info_gain = 0
    feature_sets = list(examples[0].keys())
    feature_sets.remove(target)
    for feature in feature_sets:
        feature_values = [example[feature] for example in examples]
        set_feature_value = set(feature_values)  # set无重复的属性特征值
        new_entropy = 0.0
        for value in set_feature_value:
            subset = split_data(examples, feature, value)
            prob = len(subset) / float(len(examples))  # 即p(t)
            new_entropy += prob * calc_ent(subset, target)  # 对各子集香农熵求和
        current_info_gain = baseEntropy - new_entropy
        if current_info_gain > best_info_gain:
            best_info_gain = current_info_gain
            best_feature = feature
    # print("best_feature = " + str(best_feature))
    return best_feature  # 返回特征值


# find the most common feature--应该是此时已经到达叶子节点了，但是需要确定划分了class种类，
def choose_most_feature(examples, target):
    feature_count = {}
    for example in examples:
        current_attribute = example[target]
        if current_attribute not in feature_count.keys():
            feature_count[current_attribute] = 0
        feature_count[current_attribute] += 1
    most_feature_value = max(feature_count.items(), key=lambda x: x[1])[0]
    # for vote in examples:
    #     if vote not in feature_count.keys():
    #         feature_count[vote] = 0
    #     feature_count[vote] += 1
    # sortedClassCount = sorted(feature_count.items, key=operator.itemgetter(1), reversed=True)
    return most_feature_value


def build_ID3(examples, target, labels, default):
    if not examples:
        # if intial trainning set is null, return root node
        node = Node()
        node.label = 'NULL'
        return node

    feature_values = [example[target] for example in examples]
    if feature_values.count(feature_values[0]) == len(feature_values):
        node = Node()
        node.label = feature_values[0]
        return node

    if len(examples[0]) == 1:
        node = Node()
        node.label = choose_most_feature(examples, target)
        return node

    best_feature = choose_best_feature(examples, target)
    node = Node()
    node.label = best_feature
    # give this labels to subtree--应该给一个深拷贝的对象给下一步
    labels.remove(best_feature)
    feature_values = [example[best_feature] for example in examples]
    set_feature_value = set(feature_values)
    for value in set_feature_value:
        sub_labels = labels[:]
        sub_examples = split_data(examples, best_feature, value)
        sub_node = build_ID3(sub_examples, target, sub_labels, default)
        node.children[str(value)] = sub_node
    return node


def ID3(examples, default):
    '''
    Takes in an array of examples, and returns a tree (an instance of Node)
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''
    labels = list(examples[0].keys())
    return build_ID3(examples, 'Class', labels, default)
    # PlayTennis
    # Class



def prune(node, examples):
    '''
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''


def test(node, examples):
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''


def evaluate(node, example):
    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''

    if len(node.children) == 0:
        return node.label
    feature = node.label
    value = str(example[feature])
    sub_node = node.children[value]
    return evaluate(sub_node, example)

