import torch
from torch import nn
from torch.nn.utils import rnn


class OligoMLP(nn.Module):
    """Base MLP class that geenrates the duplexing score predictions from the One-Hot encodings of the two sequences and additional features sucha as the GC content and the melting temperature difference. 
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], act_function: nn.Module = nn.ReLU()) -> None:
        """Base MLP class that geenrates the duplexing score predictions from the One-Hot encodings of the two sequences and additional features sucha as the GC content and the melting temperature difference. 

        Args:
            input_dim (int): DImension of the input ( 4*seq length + 4*seq_length + add_features)
            hidden_dims (list[int]): List wiht the hidden dimensions of each layer of the network, the length of the list represnts the number of layers.
            act_function (nn.Module, optional): Activation function used in the network. Defaults to nn.ReLU().
        """
        super().__init__()
        dimensions = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dimensions)-1): #last layer is added later
            layers.extend([nn.Linear(in_features=dimensions[i], out_features=dimensions[i+1]), act_function])
            # TODO: add regularization
        layers.extend([nn.Linear(in_features=dimensions[-1], out_features=1)]) # last layer
        self.mlp = nn.Sequential(*layers)
        self.double()


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Predicts the duplexing probability given the One-Hot encodings of the two sequences and their additional features.

        Args:
            input (torch.Tensor): One-Hot encoding of the two sequences concatnated with their additional features, dim: (batch_size, 4*seq length + 4*seq_length + add_features)

        Returns:
            torch.Tensor: Duplexing score of the two sequences.
        """
        return self.mlp(input).flatten()
    



class OligoRNN(nn.Module):
    """Recurrent Neural Network to predict the duplexing score between two sequences. The sequences are encoded and fed into the network nucleotide by nucleotide. 
    Concretely at each step the Network recieves in input two correstpoing nucleotides of the two strands. The hidden state at each time stamp is then processed by MLP
    shared across all the steps and the uoutput is pooled
    """

    def __init__(self, input_size: int, features: int, hidden_size: int, nonlinearity: str = 'tanh', pool: str = 'max', act=torch.nn.ReLU(), dropout=0) -> None:
        super().__init__()
        self.hidden_size= hidden_size
        self.pool = pool
        self.recurrent_block = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, nonlinearity=nonlinearity, dropout=dropout)
        self.shared_MLP = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size), 
            act, 
            nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=hidden_size, out_features=hidden_size),
        )
        self.features_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=features, out_features=features), 
            act, 
            nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=features, out_features=features),
        )
        self.final_MLP = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size + features, out_features=hidden_size), 
            act, 
            nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=hidden_size, out_features=1),
        )
        self.double()


    def forward(self, sequences, features):
        hidden_states = self.recurrent_block(sequences)[0]
        # vectorize the porcess of all the hidden states of the batch
        processed_hidden_states = self.shared_MLP(hidden_states.data) 
        # create a new PackedSeequence class and unpack it
        processed_hidden_states = rnn.PackedSequence(data=processed_hidden_states, batch_sizes=hidden_states.batch_sizes) 
        processed_hidden_states = rnn.unpack_sequence(processed_hidden_states)
        # pool the hidden states
        if self.pool == "max":
            pool_function = lambda h: torch.max(input=h, dim=0, keepdim=True)[0] # the output is a tuple of tensors
        elif self.pool == "min":
            pool_function = lambda h: torch.min(input=h, dim=0, keepdim=True)[0]
        elif self.pool == "mean":
            pool_function = lambda h: torch.mean(input=h, dim=0, keepdim=True)
        elif self.pool == "sum":
            pool_function = lambda h: torch.sum(input=h, dim=0, keepdim=True)
        else:
            Warning(f"The pooling function {self.pool} is not supported.") #change warning type
        pooled_hidden_states = torch.cat([pool_function(h) for h in processed_hidden_states], dim=0)
        processed_features = self.features_mlp(features)
        # generate the final prediction
        return self.final_MLP(torch.cat([pooled_hidden_states, processed_features], dim=1)).flatten()
    



class OligoLSTM(OligoRNN):


    def __init__(self, input_size: int, features: int, hidden_size: int, pool: str = 'max', act=torch.nn.ReLU(), dropout=0) -> None:
        super().__init__(input_size=input_size, features=features, hidden_size=hidden_size, pool=pool, act=act, dropout=dropout)
        self.recurrent_block = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
        self.double()