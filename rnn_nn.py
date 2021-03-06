import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , h

class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
        ############################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes      #
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h and c as 0 if these values are not given.                    #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
            
    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq=[]
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        pass

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , (h, c)
    

class RNN_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28, hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()

        ############################################################################
        #  TODO: Build a RNN classifier                                            #
        ############################################################################

        self.n_inputs = input_size
        self.n_outputs = classes

        self.rnn = nn.RNN(self.n_inputs, hidden_size)

        self.FC = nn.Linear(hidden_size, self.n_outputs)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def forward(self, x):
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################

        output, h_n = self.rnn(x)
        out = self.FC(h_n)
        return out.view(-1, self.n_outputs)


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class LSTM_Classifier(torch.nn.Module):

    def __init__(self, classes=10, input_size=28, hidden_size=128):

        super(LSTM_Classifier, self).__init__()

        ############################################################################
        #  TODO: Build a LSTM classifier                                           #
        ############################################################################
        self.n_outputs = classes

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.fc = nn.Linear(hidden_size, classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):

        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################

        output, (h_n, c_s) = self.lstm(x)
        out = self.fc(h_n)
        return out.view(-1, self.n_outputs)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)