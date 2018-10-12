import math

from node import Node


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


# find the most entropy and return the feature name
def choose_best_feature(examples, target):
    # Entropy
    baseEntropy = calc_ent(examples, target)
    best_info_gain = 0
    best_feature = -1
    current_info_gain = 0
    feature_sets = list(examples[0].keys())
    feature_sets.remove(target)
    for feature in feature_sets:
        feature_values = [example[feature] for example in examples]
        set_feature_value = set(feature_values)
        new_entropy = 0.0
        for value in set_feature_value:
            subset = split_data(examples, feature, value)
            prob = len(subset) / float(len(examples))
            new_entropy += prob * calc_ent(subset, target)
        current_info_gain = baseEntropy - new_entropy
        if current_info_gain > best_info_gain:
            best_info_gain = current_info_gain
            best_feature = feature
    # print("best_feature = " + str(best_feature))
    return best_feature


# find the most common feature
def choose_most_feature(examples, target):
    feature_count = {}
    for example in examples:
        current_attribute = example[target]
        if current_attribute not in feature_count.keys():
            feature_count[current_attribute] = 0
        feature_count[current_attribute] += 1
    most_feature_value = max(feature_count.items(), key=lambda x: x[1])[0]
    return most_feature_value


def build_ID3(examples, target, labels, default):
    if not examples:
        node = Node()
        node.label = 'NULL'
        return node

    feature_values = [example[target] for example in examples]
    if feature_values.count(feature_values[0]) == len(feature_values):
        node = Node()
        node.label = feature_values[0]
        node.sample = examples
        return node

    if len(examples[0]) == 1:
        node = Node()
        node.label = choose_most_feature(examples, target)
        node.sample = examples
        return node

    best_feature = choose_best_feature(examples, target)
    node = Node()
    node.label = best_feature
    node.sample = examples
    # give this labels to subtree
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


def build_prune(node, root, examples, set_feature_value):
    # get to the end
    if node.label in set_feature_value:
        return
    is_leaf_parent = 1
    for label in node.children:
        if not (node.children[label].label in set_feature_value):
            is_leaf_parent = 0
            break

    # find leaves' parent node
    if is_leaf_parent == 1:
        previous_accuracy = test(root, examples)
        # get a copy
        temp_node = Node()
        temp_node.label = node.label
        # for label in node.children:
        #     temp_node.children[label] = node.children[label]

        most_feature = choose_most_feature(node.sample, "Class")
        # print(most_feature)
        node.label = most_feature
        # 暂时删除节点
        # node.children = {}
        current_accuracy = test(root, examples)
        if current_accuracy <= previous_accuracy:
            node.label = temp_node.label
        else:
            # Means this leaf is cut, do pruning again
            build_prune(root, root, examples, set_feature_value)
            # print("Current : %d  Previous: %d", (current_accuracy, previous_accuracy))
    else:
        for label in node.children:
            build_prune(node.children[label], root, examples, set_feature_value)


def prune(node, examples):
    '''
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''
    feature_values = [data['Class'] for data in node.sample]
    # There is a small problem here, what if the validation set is too small that do not have enough "Class"
    # feature_values = [example['Class'] for example in examples]
    set_feature_value = set(feature_values)
    build_prune(node, node, examples, set_feature_value)


def test(node, examples):
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''
    count = 0
    for example in examples:
        if evaluate(node, example) == example['Class']:
            count += 1

    accuracy = float(count / len(examples))
    return accuracy


def do_evaluate(node, example, set_feature_value):
    # find the leaf
    if node.label in set_feature_value:
        # print(node.label)
        return node.label
    feature = node.label
    # print(set_feature_value)
    # print(feature)
    value = str(example[feature])
    if value in node.children:
        sub_node = node.children[value]
        # return evaluate(sub_node, example)
        return do_evaluate(sub_node, example, set_feature_value)
    else:
        return None


# It seems that when the test/validation set is pretty tricky, decision tree cannot cover all of the situation,
# sometimes program will encounter exception
def evaluate(node, example):
    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''
    feature_values = [data['Class'] for data in node.sample]
    set_feature_value = set(feature_values)
    return do_evaluate(node, example, set_feature_value)
