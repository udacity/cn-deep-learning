import numpy as np

class NeuralNetwork(object):
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate
        
        # Activation function is the sigmoid function
        self.activation_function = self.sigmoid
    
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array, column vector
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        ### Forward pass ###
        #Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs
        
        #### Implement the backward pass here ####
        ### Backward pass ###
        
        # 1 is the gradient of f'(x) where f(x) = x 
        output_delta = (targets - final_outputs) * 1

        hidden_delta = np.dot(self.weights_hidden_to_output.T, output_delta) * hidden_outputs * (1-hidden_outputs)
        
        # TODO: Update the weights
        self.weights_hidden_to_output += self.lr * np.dot(output_delta, hidden_outputs.T)
        self.weights_input_to_hidden += self.lr * np.dot(hidden_delta, inputs.T)
 
        
    #predict with a inputs_list
    def run(self, inputs_list): 
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        #Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs
        
        return final_outputs
