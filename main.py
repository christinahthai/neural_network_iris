"""Assignment 6 Complete Neural Network for course CS 3B with Eric Reed. This
program is the framework for loading up a dataset and divides it into
training and testing sets on the iris dataset. Serializes the NNData class
objects """

import unittest
from enum import Enum
import json
import numpy as np
import random
import collections
from abc import ABC, abstractmethod
import math


class DataMismatchError(Exception):
    """Check if there are equal quantities of labels as features in the
    test and training data"""

    def __init__(self, message):
        self.message = message


class LayerType(Enum):
    """Specify elements to be INPUT, HIDDEN, OR OUTPUT"""
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class MultiLinkNode(ABC):
    class Side(Enum):
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        # Binary encoding to track neighboring nodes when information
        # is available
        self._reporting_nodes = dict.fromkeys([MultiLinkNode.Side.UPSTREAM,
                                               MultiLinkNode.Side.DOWNSTREAM],
                                              0)
        # Represent the reporting node values as binary encoding when
        # all nodes have been reported
        self._reference_value = dict.fromkeys([MultiLinkNode.Side.UPSTREAM,
                                               MultiLinkNode.Side.DOWNSTREAM],
                                              0)
        # References to neighboring nodes upstream and downstream
        self._neighbors = dict.fromkeys([MultiLinkNode.Side.UPSTREAM,
                                         MultiLinkNode.Side.DOWNSTREAM], [])

    def __str__(self):
        all_nodes = list()
        for key, value in self._neighbors.items():
            for v in value:
                all_nodes.append([key, id(v)])
        node_list_upstream = [i[1] for i in all_nodes if i[0] ==
                              MultiLinkNode.Side.UPSTREAM]
        node_list_downstream = [i[1] for i in all_nodes if i[0] ==
                                MultiLinkNode.Side.DOWNSTREAM]
        return f'ID upstream neighboring nodes: {node_list_upstream} \n' \
               f'ID downstream neighboring nodes: {node_list_downstream} \n' \
               f'ID of current node {id(self)}'

    @abstractmethod
    def _process_new_neighbor(self, node, side):
        """Update reference node to be added as a key in the self._weights
        dictionary"""
        pass

    def reset_neighbors(self, nodes: list, side: Side):
        """Resets the nodes that link into this node either upstream or
        downstream. Copies node parameters into appropriate entry of
        neighbors and processes a new neighbor for each node. Then
        calculates and stores the appropriate value in the correct element
        of the reference value list"""
        self._neighbors[side] = nodes.copy()
        for node in nodes:
            self._process_new_neighbor(node, side)
        self._reference_value[side] = (1 << len(nodes)) - 1


class NNData:
    class Order(Enum):
        """Define whether the training data is presented in random or
        sequential order"""
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        """Define if the set is testing data set or the training data
        set"""
        TRAIN = 0
        TEST = 1

    @staticmethod
    def percentage_limiter(percentage: float):
        """Limits training factor percentage between 0-1"""
        if percentage < 0:
            percentage = 0
        if percentage > 1:
            percentage = 1
        return float(percentage)

    def __init__(self, features=None, labels=None, train_factor=0.9):
        self._train_factor = NNData.percentage_limiter(train_factor)
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._features = None
        self._labels = None
        self._train_indices = []
        self._test_indices = []
        self._train_pool = collections.deque()
        self._test_pool = collections.deque()
        try:
            self.load_data(features, labels)
            self.split_set()
        except (ValueError, DataMismatchError):
            pass

    def load_data(self, features, labels):
        """Loads features/labels data into multidimensional array"""
        if features is None or labels is None:
            self._features = None
            self._labels = None
            return
        if len(features) != len(labels):
            raise DataMismatchError("Label and example lists have "
                                    "different lengths")
        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._features = None
            self._labels = None
            raise ValueError("Label and example lists must be homogeneous "
                             "and numeric lists of lists")

    def split_set(self, new_train_factor=None):
        """Split training and testing data sets by setting new training
        set factor"""
        if new_train_factor is None:
            new_train_factor = self._train_factor
        else:
            self._train_factor = self.percentage_limiter(new_train_factor)
        sample_size = len(self._features)
        training_size = int(round(sample_size * self._train_factor, 0))
        self._train_indices = random.sample([i for i in range(
            sample_size)], k=training_size)
        self._test_indices = [i for i in range(sample_size) if i not in
                              self._train_indices]

    def prime_data(self, target_set=None, order=None):
        """Copy train/test indices into a train/test pool and order them
        sequentially or randomly"""
        if target_set is None:
            self._train_pool = self._train_indices.copy()
            self._test_pool = self._test_indices.copy()
        elif target_set == NNData.Set.TEST:
            self._test_pool = self._test_indices.copy()
        elif target_set == NNData.Set.TRAIN:
            self._train_pool = self._train_indices.copy()
        if order == NNData.Order.RANDOM:
            random.shuffle(self._train_pool)
            random.shuffle(self._test_pool)
        self._train_pool = collections.deque(self._train_pool)
        self._test_pool = collections.deque(self._test_pool)

    def get_one_item(self, target_set=None):
        """Gather one pair of train and test values"""
        if target_set == NNData.Set.TRAIN or target_set is None:
            index = self._train_pool.popleft()
            feature_value = self._features[index]
            label_value = self._labels[index]
        elif target_set == NNData.Set.TEST:
            index = self._test_pool.popleft()
            feature_value = self._features[index]
            label_value = self._labels[index]
        return (feature_value, label_value)

    def number_of_samples(self, target_set=None):
        """Calculate the sample size of the specified set (train/test)"""
        if target_set == NNData.Set.TRAIN:
            target_set_sample_size = len(self._features)
        elif target_set == NNData.Set.TEST:
            target_set_sample_size = len(self._features)
        else:
            target_set_sample_size = len(self._features)
        return target_set_sample_size

    def pool_is_empty(self, target_set=None):
        """Return false if the target set (test/train) is empty"""
        empty_set = True
        if target_set == NNData.Set.TRAIN or target_set is None:
            target_set = self._train_pool
        elif target_set == NNData.Set.TEST:
            target_set = self._test_pool
        if bool(target_set):
            empty_set = False
        return empty_set


class Neurode(MultiLinkNode):

    def __init__(self, node_type, learning_rate=.05):
        super().__init__()
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}

    def _process_new_neighbor(self, node, side: MultiLinkNode.Side):
        """Updates weight dictionary if UPSTREAM node is generated with a
        randomly generated float between 0 and 1 when a new node is added"""
        if side is MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.random()

    def _check_in(self, node, side: MultiLinkNode.Side):
        """Check in upstream nodes that have information to update the
        following node with"""
        node_number = self._neighbors[side].index(node)
        self._reporting_nodes[side] = \
            self._reporting_nodes[side] | 1 << node_number
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        else:
            return False

    @property
    def value(self):
        return self._value

    @property
    def node_type(self):
        return self._node_type

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self._learning_rate = learning_rate

    def get_weight(self, node):
        """Get weights of previously passed upstream nodes relative to
        current node"""
        return self._weights[node]


class FFNeurode(Neurode):

    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        """Result of sigmoid function at value"""
        return 1 / (1 + np.exp(-value))

    def _calculate_value(self):
        """Calculate the weighted sum of the upstream nodes' values and
        pass through the sigmoid function"""
        input_sum = 0
        for node, weight in self._weights.items():
            input_sum += node.value * weight
        self._value = self._sigmoid(input_sum)

    def _fire_downstream(self):
        """Send values upstream"""
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, from_node):
        """Check if node has upstream node data ready to send to the
        downstream node"""
        if self._check_in(from_node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def set_input(self, input_value: float):
        """Set value of input layer node"""
        self._value = input_value
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)


class BPNeurode(Neurode):

    def __init__(self, my_type):
        super().__init__(my_type)
        self._delta = 0

    @property
    def delta(self):
        return self._delta

    @staticmethod
    def _sigmoid_derivative(value):
        return value * (1.0 - value)

    def _calculate_delta(self, expected_value=None):
        """Calculate the delta for output, hidden, input nodes"""
        if self._node_type == LayerType.OUTPUT:
            error = expected_value - self.value
            self._delta = error * self._sigmoid_derivative(self.value)
        else:
            self._delta = 0
            for neurode in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                self._delta += neurode.get_weight(self) * neurode.delta
            self._delta *= self._sigmoid_derivative(self.value)

    def set_expected(self, expected_value: float):
        self._calculate_delta(expected_value)
        self._fire_upstream()

    def data_ready_downstream(self, from_node):
        """Check if data is ready downstream and collect the data to
        make it available in the next layer up after calculating the
        current node's delta value, and update the weights"""
        if self._check_in(from_node, MultiLinkNode.Side.DOWNSTREAM):
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def adjust_weights(self, node, adjustment):
        """Set strength of node importance by using node reference to
        add adjustments to appropriate entry of weights """
        self._weights[node] += adjustment

    def _update_weights(self):
        """Iterate through downstream neighbors and use adjust_weights
        method to request an adjustment to weight/significance of
        current node's data"""
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adjustment = node.learning_rate * node.delta * self.value
            node.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        for node in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            node.data_ready_downstream(self)


class FFBPNeurode(FFNeurode, BPNeurode):
    pass


class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    """Class makes current the head node when first item is added."""

    class EmptyListError(Exception):
        """Check if list is empty and return custom error message if list
        is empty"""

        def __init__(self, message):
            self.message = message

    def __init__(self):
        self._head = None
        self._curr = None
        self._tail = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr and self._curr.next:
            ret_val = self._curr.data
            self._curr = self._curr.next
            return ret_val
        raise StopIteration

    def move_forward(self):
        """Check if current node is empty or the tail node. Set new current
        node to be the next one in the doublylinkedlist"""
        if not self._curr:
            raise DoublyLinkedList.EmptyListError("Empty list - cannot "
                                                  "move to next object")
        # If there is another node after the current node, set the next
        # node to be the current node, otherwise the current node is the tail
        # and raise an Index Error
        if self._curr.next:
            self._curr = self._curr.next
        else:
            raise IndexError

    def move_back(self):
        """Check if the current node is empty or the tail node. If neither,
        then set the new current node to the next node in the doubly linked
        list"""
        if not self._curr:
            raise DoublyLinkedList.EmptyListError("List empty - cannot move "
                                                  "back")
        if self._curr.prev:
            self._curr = self._curr.prev
        else:
            raise IndexError

    def reset_to_head(self):
        """Set the current node to be the head node"""
        if not self._head:
            raise DoublyLinkedList.EmptyListError("Empty list - cannot "
                                                  "reset to head")
        self._curr = self._head

    def reset_to_tail(self):
        """Set current node to be the tail node"""
        if not self._tail:
            raise DoublyLinkedList.EmptyListError("Empty list")
        self._curr = self._tail

    def add_to_head(self, data):
        """Add a new node to the doubly linked list at the head node
        position if there already exists a head. If there is only one node,
        set the head equal to the tail"""
        new_node = Node(data)
        new_node.next = self._head
        if self._head:
            self._head.prev = new_node
        self._head = new_node
        if self._tail is None:
            self._tail = new_node
        self.reset_to_head()

    def add_after_cur(self, data):
        """Add a new node after the current node."""
        if not self._curr:
            raise self.EmptyListError
        new_node = Node(data)
        new_node.prev = self._curr
        new_node.next = self._curr.next
        if self._curr.next:
            self._curr.next.prev = new_node
        self._curr.next = new_node
        if self._tail == self._curr:
            self._tail = new_node

    def remove_from_head(self):
        """Remove the head node in the doubly linked list"""
        if not self._head:
            raise DoublyLinkedList.EmptyListError("Nothing to remove from "
                                                  "head of empty list")
        ret_val = self._head.data
        self._head = self._head.next
        if self._head:
            self._head.prev = None
        else:
            self._tail = None
        self.reset_to_head()
        return ret_val

    def remove_after_cur(self):
        """Remove the node that is directly after the current node"""
        if not self._curr:
            raise DoublyLinkedList.EmptyListError("No data to remove in "
                                                  "remove_after_cur")
        # Cannot delete node after tail node
        if self._curr.next is None:
            raise IndexError
        ret_val = self._curr.next.data
        # If current node is second to last, delete last node
        if self._curr.next == self._tail:
            self._tail = self._curr
            self._curr.next = None
            print("Deleting last node")
        # Delete node after current node
        else:
            self._curr.next = self._curr.next.next
            self._curr.next.prev = self._curr
        return ret_val

    def get_current_data(self):
        """View the current node's data"""
        if not self._curr:
            raise DoublyLinkedList.EmptyListError("Dataset Empty in "
                                                  "get_current_data")
        return self._curr.data


class LayerList(DoublyLinkedList, FFBPNeurode):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        input_nodes = []
        output_nodes = []
        # Randomly generate input and output FFBPNeurode objects according
        # to user input/output values
        for _ in range(inputs):
            input_nodes.append(FFBPNeurode(LayerType.INPUT))
        for _ in range(outputs):
            output_nodes.append(FFBPNeurode(LayerType.OUTPUT))
        # Link the input and output layers together
        for node in input_nodes:
            node.reset_neighbors(output_nodes,
                                 MultiLinkNode.Side.DOWNSTREAM)
        for node in output_nodes:
            node.reset_neighbors(input_nodes, MultiLinkNode.Side.UPSTREAM)
        # Add FFBPNeurode objects into the doubly linked list
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self.add_to_head(self._input_nodes)
        self.add_after_cur(self._output_nodes)
        self._num_layers = 2

    def add_layer(self, num_nodes: int):
        """Add hidden layer stored in a node after the current layer"""
        if self._curr == self._tail:
            raise IndexError
        hidden_nodes = []
        for _ in range(num_nodes):
            hidden_nodes.append(FFBPNeurode(LayerType.HIDDEN))
        # There is only the head and tail nodes in the DoublyLinkedList
        if self._num_layers == 2:
            for neurode in hidden_nodes:
                neurode.reset_neighbors(self._input_nodes,
                                        MultiLinkNode.Side.UPSTREAM)
                neurode.reset_neighbors(self._output_nodes,
                                        MultiLinkNode.Side.DOWNSTREAM)
            self.add_after_cur(hidden_nodes)
        else:
            for neurode in hidden_nodes:
                neurode.reset_neighbors(self.get_current_data(),
                                        MultiLinkNode.Side.UPSTREAM)
            self.add_after_cur(hidden_nodes)
            self.move_forward()
            for neurode in hidden_nodes:
                neurode.reset_neighbors(self.get_current_data(),
                                        MultiLinkNode.Side.DOWNSTREAM)
            self.move_back()
        self._num_layers += 1

    def remove_layer(self):
        """Remove the layer AFTER the current layer in the linked list node
        - but do not let client remove output layer"""
        if self._curr.next == self._tail:
            raise IndexError
        else:
            self.remove_after_cur()
            self._num_layers -= 1

    @property
    def input_nodes(self):
        return self._input_nodes

    @property
    def output_nodes(self):
        return self._output_nodes


def load_XOR():
    """Load mock XOR data for features and labels sets"""
    features = [[0, 0], [1, 0], [0, 1], [1, 1]]
    labels = [[0], [1], [1], [0]]
    data = NNData(features, labels, 1)
    return data


class FFBPNetwork(LayerList):
    """Added functionality to remove a hidden layer, browser the network,
    and change the learning rate"""

    class EmptySetException(Exception):
        """Check if list is empty and return custom error message if list
        is empty"""
        def __init__(self, message):
            self.message = message

    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__(num_inputs, num_outputs)
        self.layers = LayerList(num_inputs, num_outputs)
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs

    def add_hidden_layer(self, num_nodes: int, position=0):
        """Add a new hidden layer based on user input of position.
        Default 0 implies hidden layer is given right after the
        input layer"""
        self.reset_to_head()
        for _ in range(position):
            self.move_forward()
        self.layers.add_layer(num_nodes)

    def train(self, data_set: NNData, epochs=1000, verbosity=2,
              order=NNData.Order.RANDOM):
        if data_set.number_of_samples(NNData.Set.TRAIN) == 0:
            raise FFBPNetwork.EmptySetException("No data in training set")
        for epoch in range(0, epochs):
            data_set.prime_data(order=order)
            sum_error = 0
            while not data_set.pool_is_empty(NNData.Set.TRAIN):
                x, y = data_set.get_one_item(NNData.Set.TRAIN)
                for j, node in enumerate(self.layers.input_nodes):
                    node.set_input(x[j])
                produced = []
                for j, node in enumerate(self.layers.output_nodes):
                    node.set_expected(y[j])
                    sum_error += (node.value - y[j]) ** 2 / self._num_outputs
                    produced.append(node.value)

                if epoch % 1000 == 0 and verbosity > 1:
                    print("Sample", x, "expected", y, "produced", produced)
            if epoch % 100 == 0 and verbosity > 0:
                print("Epoch", epoch, "RMSE = ", math.sqrt(
                    sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))
        print("Final Epoch RMSE = ", math.sqrt(
            sum_error / data_set.number_of_samples(NNData.Set.TRAIN)))

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        if data_set.number_of_samples(NNData.Set.TEST) == 0:
            raise FFBPNetwork.EmptySetException
        data_set.prime_data(order=order)
        sum_error = 0
        while not data_set.pool_is_empty(NNData.Set.TEST):
            x, y = data_set.get_one_item(NNData.Set.TEST)
            for j, node in enumerate(self.layers.input_nodes):
                node.set_input(x[j])
            produced = []
            for j, node in enumerate(self.layers.output_nodes):
                sum_error += (node.value - y[j]) ** 2 / self._num_outputs
                produced.append(node.value)

            print(x, ",", y, ",", produced)
        print("RMSE = ", math.sqrt(
            sum_error / data_set.number_of_samples(NNData.Set.TEST)))

    # def train(self, data_set: NNData, epochs=1000, verbosity=2,
    #           order=NNData.Order.RANDOM):
    #     """Prime data using the specified order argument, and set them
    #     as the input nodes to run the neural network on each set. Also
    #     check expected values and calculate the RMSE (error) """
    #
    #     if data_set.number_of_samples(NNData.Set.TRAIN) == 0:
    #         raise self.EmptySetException("No training data set")
    #     else:
    #         print("Training data... \n")
    #     for _ in range(epochs):
    #         errs = []
    #         # Prime data based on random order, or sequential
    #         data_set.prime_data(NNData.Set.TRAIN, order)
    #         rsme = 0
    #         while not data_set.pool_is_empty(NNData.Set.TRAIN):
    #             # fetch one training data pair e.g. [inputs, actual_output]
    #             feature, label = data_set.get_one_item(NNData.Set.TRAIN)
    #
    #             for i in range(len(self.layers.input_nodes)):
    #                 self.layers.input_nodes[i].set_input(
    #                     input_value=feature[i])
    #             predicted_value = [node.value for node in
    #                                self.layers.output_nodes]
    #             if verbosity > 1 and _ == 1000:
    #                 print(predicted_value, label)
    #             # Calculate error from predicted and actual using RMSE method
    #             for i in range(len(predicted_value)):
    #                 errs.append((predicted_value[i] - label[i]) ** 2)
    #             # provide the label to the output layer neurodes
    #             for i in range(len(self.layers.output_nodes)):
    #                 self.layers.output_nodes[i].set_expected(
    #                     expected_value=label[i])
    #         rmse = (sum(errs) / len(errs)) ** 0.5
    #         if verbosity > 0 and _ % 100 == 0:
    #             print(rmse)
    #     return rmse
    #
    # def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
    #     """Test method for predicting iris classification for test data set"""
    #     if data_set.number_of_samples(NNData.Set.TEST) == 0:
    #         raise self.EmptySetException("No test data set")
    #     else:
    #         print("Testing data... \n")
    #     errs = []
    #     data_set.prime_data(data_set.Set.TEST, order)
    #     while data_set.Set.TEST:
    #         try:
    #             test_pair = data_set.get_one_item(NNData.Set.TEST)
    #         except IndexError:
    #             break
    #         feature = test_pair[0]
    #         label = test_pair[1]
    #         for i in range(len(self.layers.input_nodes)):
    #             self.layers.input_nodes[i].set_input(
    #                 input_value=feature[i])
    #
    #         predicted_value = [node.value for node in
    #                            self.layers.output_nodes]
    #
    #         print(f"Feature: ", feature, "Label: ", label, "Prediction: ",
    #               predicted_value)
    #         for i in range(len(predicted_value)):
    #             errs.append((predicted_value[i] - label[i]) ** 2)
    #     return (sum(errs) / len(errs)) ** -0.5


def run_iris():
    network = FFBPNetwork(4, 3)
    network.add_hidden_layer(3)
    Iris_X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
              [4.6, 3.1, 1.5, 0.2],
              [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3],
              [5, 3.4, 1.5, 0.2],
              [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2],
              [4.8, 3.4, 1.6, 0.2],
              [4.8, 3, 1.4, 0.1], [4.3, 3, 1.1, 0.1], [5.8, 4, 1.2, 0.2],
              [5.7, 4.4, 1.5, 0.4],
              [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3],
              [5.1, 3.8, 1.5, 0.3],
              [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1, 0.2],
              [5.1, 3.3, 1.7, 0.5],
              [4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 0.2], [5, 3.4, 1.6, 0.4],
              [5.2, 3.5, 1.5, 0.2],
              [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2],
              [5.4, 3.4, 1.5, 0.4],
              [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1],
              [5, 3.2, 1.2, 0.2],
              [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3, 1.3, 0.2],
              [5.1, 3.4, 1.5, 0.2],
              [5, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3], [4.4, 3.2, 1.3, 0.2],
              [5, 3.5, 1.6, 0.6],
              [5.1, 3.8, 1.9, 0.4], [4.8, 3, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2],
              [4.6, 3.2, 1.4, 0.2],
              [5.3, 3.7, 1.5, 0.2], [5, 3.3, 1.4, 0.2], [7, 3.2, 4.7, 1.4],
              [6.4, 3.2, 4.5, 1.5],
              [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5],
              [5.7, 2.8, 4.5, 1.3],
              [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1], [6.6, 2.9, 4.6, 1.3],
              [5.2, 2.7, 3.9, 1.4], [5, 2, 3.5, 1],
              [5.9, 3, 4.2, 1.5], [6, 2.2, 4, 1], [6.1, 2.9, 4.7, 1.4],
              [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
              [5.6, 3, 4.5, 1.5], [5.8, 2.7, 4.1, 1], [6.2, 2.2, 4.5, 1.5],
              [5.6, 2.5, 3.9, 1.1],
              [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4, 1.3], [6.3, 2.5, 4.9, 1.5],
              [6.1, 2.8, 4.7, 1.2],
              [6.4, 2.9, 4.3, 1.3], [6.6, 3, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4],
              [6.7, 3, 5, 1.7], [6, 2.9, 4.5, 1.5],
              [5.7, 2.6, 3.5, 1], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1],
              [5.8, 2.7, 3.9, 1.2],
              [6, 2.7, 5.1, 1.6], [5.4, 3, 4.5, 1.5], [6, 3.4, 4.5, 1.6],
              [6.7, 3.1, 4.7, 1.5],
              [6.3, 2.3, 4.4, 1.3], [5.6, 3, 4.1, 1.3], [5.5, 2.5, 4, 1.3],
              [5.5, 2.6, 4.4, 1.2],
              [6.1, 3, 4.6, 1.4], [5.8, 2.6, 4, 1.2], [5, 2.3, 3.3, 1],
              [5.6, 2.7, 4.2, 1.3], [5.7, 3, 4.2, 1.2],
              [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3, 1.1],
              [5.7, 2.8, 4.1, 1.3],
              [6.3, 3.3, 6, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3, 5.9, 2.1],
              [6.3, 2.9, 5.6, 1.8],
              [6.5, 3, 5.8, 2.2], [7.6, 3, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7],
              [7.3, 2.9, 6.3, 1.8],
              [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2],
              [6.4, 2.7, 5.3, 1.9],
              [6.8, 3, 5.5, 2.1], [5.7, 2.5, 5, 2], [5.8, 2.8, 5.1, 2.4],
              [6.4, 3.2, 5.3, 2.3], [6.5, 3, 5.5, 1.8],
              [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6, 2.2, 5, 1.5],
              [6.9, 3.2, 5.7, 2.3],
              [5.6, 2.8, 4.9, 2], [7.7, 2.8, 6.7, 2], [6.3, 2.7, 4.9, 1.8],
              [6.7, 3.3, 5.7, 2.1],
              [7.2, 3.2, 6, 1.8], [6.2, 2.8, 4.8, 1.8], [6.1, 3, 4.9, 1.8],
              [6.4, 2.8, 5.6, 2.1],
              [7.2, 3, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2],
              [6.4, 2.8, 5.6, 2.2],
              [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4], [7.7, 3, 6.1, 2.3],
              [6.3, 3.4, 5.6, 2.4],
              [6.4, 3.1, 5.5, 1.8], [6, 3, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1],
              [6.7, 3.1, 5.6, 2.4],
              [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3],
              [6.7, 3.3, 5.7, 2.5],
              [6.7, 3, 5.2, 2.3], [6.3, 2.5, 5, 1.9], [6.5, 3, 5.2, 2],
              [6.2, 3.4, 5.4, 2.3], [5.9, 3, 5.1, 1.8]]
    Iris_Y = [[1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ]]
    data = NNData(Iris_X, Iris_Y, .7)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_sin():
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07],
             [0.08], [0.09], [0.1], [0.11], [0.12],
             [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2],
             [0.21], [0.22], [0.23], [0.24], [0.25],
             [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33],
             [0.34], [0.35], [0.36], [0.37], [0.38],
             [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46],
             [0.47], [0.48], [0.49], [0.5], [0.51],
             [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59],
             [0.6], [0.61], [0.62], [0.63], [0.64],
             [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72],
             [0.73], [0.74], [0.75], [0.76], [0.77],
             [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85],
             [0.86], [0.87], [0.88], [0.89], [0.9],
             [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98],
             [0.99], [1], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11],
             [1.12], [1.13], [1.14], [1.15], [1.16],
             [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24],
             [1.25], [1.26], [1.27], [1.28], [1.29],
             [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37],
             [1.38], [1.39], [1.4], [1.41], [1.42],
             [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5],
             [1.51], [1.52], [1.53], [1.54], [1.55],
             [1.56], [1.57]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331],
             [0.0299955002024957], [0.0399893341866342],
             [0.0499791692706783], [0.0599640064794446], [0.0699428473375328],
             [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175],
             [0.119712207288919],
             [0.129634142619695], [0.139543114644236], [0.149438132473599],
             [0.159318206614246],
             [0.169182349066996], [0.179029573425824], [0.188858894976501],
             [0.198669330795061], [0.2084598998461],
             [0.218229623080869], [0.227977523535188], [0.237702626427135],
             [0.247403959254523],
             [0.257080551892155], [0.266731436688831], [0.276355648564114],
             [0.285952225104836], [0.29552020666134],
             [0.305058636443443], [0.314566560616118], [0.324043028394868],
             [0.333487092140814],
             [0.342897807455451], [0.35227423327509], [0.361615431964962],
             [0.370920469412983], [0.380188415123161],
             [0.389418342308651], [0.398609327984423], [0.40776045305957],
             [0.416870802429211], [0.425939465066],
             [0.43496553411123], [0.44394810696552], [0.452886285379068],
             [0.461779175541483], [0.470625888171158],
             [0.479425538604203], [0.488177246882907], [0.496880137843737],
             [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883],
             [0.539632048733969],
             [0.548023936791874], [0.556361022912784], [0.564642473395035],
             [0.572867460100481],
             [0.581035160537305], [0.58914475794227], [0.597195441362392],
             [0.60518640573604], [0.613116851973434],
             [0.62098598703656], [0.628793024018469], [0.636537182221968],
             [0.644217687237691], [0.651833771021537],
             [0.659384671971473], [0.666869635003698], [0.674287911628145],
             [0.681638760023334],
             [0.688921445110551], [0.696135238627357], [0.70327941920041],
             [0.710353272417608], [0.717356090899523],
             [0.724287174370143], [0.731145829726896], [0.737931371109963],
             [0.744643119970859],
             [0.751280405140293], [0.757842562895277], [0.764328937025505],
             [0.770738878898969],
             [0.777071747526824], [0.783326909627483], [0.78950373968995],
             [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998],
             [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015],
             [0.852108021949363],
             [0.857298989188603], [0.862404227243338], [0.867423225594017],
             [0.872355482344986],
             [0.877200504274682], [0.881957806884948], [0.886626914449487],
             [0.891207360061435],
             [0.895698685680048], [0.900100442176505], [0.904412189378826],
             [0.908633496115883],
             [0.912763940260521], [0.916803108771767], [0.920750597736136],
             [0.92460601240802], [0.928368967249167],
             [0.932039085967226], [0.935616001553386], [0.939099356319068],
             [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516],
             [0.955100855584692],
             [0.958015860289225], [0.960835064206073], [0.963558185417193],
             [0.966184951612734],
             [0.968715100118265], [0.971148377921045], [0.973484541695319],
             [0.975723357826659],
             [0.977864602435316], [0.979908061398614], [0.98185353037236],
             [0.983700814811277], [0.98544972998846],
             [0.98710010101385], [0.98865176285172], [0.990104560337178],
             [0.991458348191686], [0.992712991037588],
             [0.993868363411645], [0.994924349777581], [0.99588084453764],
             [0.996737752043143], [0.997494986604054],
             [0.998152472497548], [0.998710143975583], [0.999167945271476],
             [0.999525830605479],
             [0.999783764189357], [0.999941720229966], [0.999999682931835]]
    data = NNData(sin_X, sin_Y, .1)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_XOR():
    # Student should replace both lines of code below
    print("Student Code is missing")
    assert False


class MultiTypeEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, collections.deque):
            return {"__deque__": list(o)}
        elif isinstance(o, np.ndarray):
            return {"__NDarray__": o.tolist()}
        elif isinstance(o, NNData):
            return {"__NNData__": o.__dict__}
        else:
            json.JSONEncoder.default(self, o)


def multi_type_decoder(o):
    """Encoding dictionary for NNData class object"""
    if "__deque__" in o:
        return collections.deque(o["__deque__"])
    if "__NDarray__" in o:
        return np.array(o["__NDarray__"])
    if "__NNData__" in o:
        ret_obj = NNData()
        dec_obj = o["__NNData__"]
        ret_obj._features = dec_obj["_features"]
        ret_obj._labels = dec_obj["_labels"]
        ret_obj._train_indices = dec_obj["_train_indices"]
        ret_obj._test_indices = dec_obj["_test_indices"]
        ret_obj._train_factor = dec_obj["_train_factor"]
        ret_obj._train_pool = dec_obj["_train_pool"]
        ret_obj._test_pool = dec_obj["_test_pool"]
        return ret_obj
    else:
        return o

xor_data = NNData()


with open("sin_data.txt", "r") as f:
    my_obj = json.load(f, object_hook=multi_type_decoder)
    print(type(my_obj))
    print(my_obj)


def main():

    xor_data = load_XOR()
    xor_data_encoded = MultiTypeEncoder().default(xor_data)
    xor_data_decoded = multi_type_decoder(xor_data_encoded)
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(34)
    network.train(xor_data_decoded, order=NNData.Order.RANDOM)
    with open("sin_data.txt", "r") as f:
        sin_decoded = json.load(f, object_hook=multi_type_decoder)
    network.train(sin_decoded, order=NNData.Order.RANDOM)


if __name__ == "__main__":
    main()

"""
--- sample run #1 ---
/Users/christinathai/PycharmProjects/cs3b_assignment1/venv/bin/python /Users/christinathai/PycharmProjects/cs3b_assignment1/main.py
<class '__main__.NNData'>
<__main__.NNData object at 0x7fbda5683d00>
Sample [1. 0.] expected [1.] produced [0.5051148240549821]
Sample [1. 1.] expected [0.] produced [0.5066609627289992]
Sample [0. 0.] expected [0.] produced [0.5]
Sample [0. 1.] expected [1.] produced [0.5]
Epoch 0 RMSE =  0.5004040039117412
Epoch 100 RMSE =  0.5003947512824637
Epoch 200 RMSE =  0.5003921510417721
Epoch 300 RMSE =  0.5003913915985478
Epoch 400 RMSE =  0.5003911460624721
Epoch 500 RMSE =  0.5003911212489031
Epoch 600 RMSE =  0.5003910942989355
Epoch 700 RMSE =  0.5003910902417169
Epoch 800 RMSE =  0.5003910877897471
Epoch 900 RMSE =  0.5003910863192532
Final Epoch RMSE =  0.5003910822090704
Sample [0.15] expected [0.14943813] produced [0.5000069890942288]
Sample [0.24] expected [0.23770263] produced [0.49997174355443474]
Sample [0.66] expected [0.61311685] produced [0.49979247157396645]
Sample [1.46] expected [0.99386836] produced [0.4998821699241625]
Sample [0.9] expected [0.78332691] produced [0.5019557857399031]
Sample [0.08] expected [0.07991469] produced [0.5002371560244238]
Sample [0.97] expected [0.82488571] produced [0.5027735603891403]
Sample [0.49] expected [0.47062589] produced [0.5018795049012693]
Sample [0.34] expected [0.33348709] produced [0.5012878783682152]
Sample [0.01] expected [0.00999983] produced [0.5000360959877175]
Sample [0.87] expected [0.76432894] produced [0.5031269873073155]
Sample [0.8] expected [0.71735609] produced [0.5034434656938128]
Sample [0.47] expected [0.45288629] produced [0.502274387806444]
Sample [0.17] expected [0.16918235] produced [0.5008103245365976]
Sample [0.61] expected [0.57286746] produced [0.5028001404647263]
Sample [0.54] expected [0.51413599] produced [0.5025509399371216]
Sample [0.21] expected [0.2084599] produced [0.50099614483303]
Sample [0.44] expected [0.42593947] produced [0.5020026831487924]
Sample [1.1] expected [0.89120736] produced [0.504891534811995]
Sample [0.69] expected [0.63653718] produced [0.5039845450757415]
Sample [1.2] expected [0.93203909] produced [0.5072722356163505]
Sample [0.39] expected [0.38018842] produced [0.502984697906106]
Sample [0.41] expected [0.39860933] produced [0.5030764002049565]
Sample [1.36] expected [0.9778646] produced [0.5100214041385135]
Sample [0.82] expected [0.73114583] produced [0.5076723028181627]
Sample [0.53] expected [0.50553334] produced [0.50526255587421]
Sample [0.83] expected [0.73793137] produced [0.5082412911899594]
Sample [1.45] expected [0.99271299] produced [0.5152576172790843]
Sample [1.48] expected [0.99588084] produced [0.5187682599642734]
Sample [0.48] expected [0.46177918] produced [0.5071470752135372]
Sample [0.56] expected [0.5311862] produced [0.5082999586372222]
Epoch 0 RMSE =  0.12712023288655422
Epoch 100 RMSE =  0.09600162855982146
Epoch 200 RMSE =  0.09599922154862002
Epoch 300 RMSE =  0.09599913201988723
Epoch 400 RMSE =  0.09599924626650834
Epoch 500 RMSE =  0.09599902828665934
Epoch 600 RMSE =  0.09599880545909735
Epoch 700 RMSE =  0.09599877811588027
Epoch 800 RMSE =  0.09599894797898946
Epoch 900 RMSE =  0.0959992363217213
Final Epoch RMSE =  0.09599898739952994

Process finished with exit code 0
"""
