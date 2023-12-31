import torch
from torch import nn
from torch.nn.utils import rnn

def parse_act_function(act_function: str):
    if act_function == 'relu':
        return nn.ReLU()
    if act_function == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"{act_function} is not supported.")


class OligoMLP(nn.Module):
    """Base MLP class that geenrates the duplexing score predictions from the One-Hot encodings of the two sequences and additional features sucha as the GC content and the melting temperature difference. 
    """

    def __init__(self, input_size: int, hidden_size: int, n_layers : int, act_function: str = "relu", dropout: float = 0) -> None:
        """Base MLP class that geenrates the duplexing score predictions from the One-Hot encodings of the two sequences and additional features sucha as the GC content and the melting temperature difference. 

        Args:
            input_size (int): DImension of the input ( 4*seq length + 4*seq_length + add_features)
            hidden_dims (list[int]): List wiht the hidden dimensions of each layer of the network, the length of the list represnts the number of layers.
            act_function (nn.Module, optional): Activation function used in the network. Defaults to nn.ReLU().
        """
        super().__init__()
        act_function = parse_act_function(act_function)
        dimensions = [input_size] + [hidden_size for i in range(n_layers)]
        layers = []
        for i in range(len(dimensions)-1): #last layer is added later
            layers.extend([nn.Linear(in_features=dimensions[i], out_features=dimensions[i+1]), act_function, nn.Dropout(p=dropout)])
            # TODO: add regularization
        layers.append(nn.Linear(in_features=dimensions[-1], out_features=1)) # last layer
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

    def __init__(
            self, 
            input_size: int, 
            features_size: int, 
            hidden_size: int, 
            hidden_size_ecoder: int, 
            n_layers: int, 
            nonlinearity: str = 'tanh', 
            pool: str = 'max', 
            act_function: str = "relu", 
            n_layers_mlp: int = 1, 
            n_layers_encoder: int = 1, 
            dropout: float = 0, 
            bidirectional: bool = False
        ) -> None:

        super().__init__()
        act_function = parse_act_function(act_function)
        hidden_size_shared= hidden_size + (bidirectional * hidden_size) # if bidirectional the output tensor has doubled length
        self.pool = pool
        self.n_layers = n_layers

        self.encoding_MLP = [torch.nn.Linear(in_features=input_size, out_features=hidden_size_ecoder), act_function, nn.Dropout(p=dropout)]
        for _ in range(n_layers_encoder-1):
            self.encoding_MLP.extend([torch.nn.Linear(in_features=hidden_size_ecoder, out_features=hidden_size_ecoder), act_function, nn.Dropout(p=dropout)])
        self.encoding_MLP.append(torch.nn.Linear(in_features=hidden_size_ecoder, out_features=hidden_size_ecoder))
        self.encoding_MLP = nn.Sequential(*self.encoding_MLP)

        self.recurrent_block = torch.nn.RNN(input_size=hidden_size_ecoder, hidden_size=hidden_size, num_layers=n_layers, nonlinearity=nonlinearity, dropout=dropout, bidirectional=bidirectional)

        self.shared_MLP = []
        for _ in range(n_layers_mlp):
            self.shared_MLP.extend([torch.nn.Linear(in_features=hidden_size_shared, out_features=hidden_size_shared), act_function, nn.Dropout(p=dropout)])
        self.shared_MLP.append(torch.nn.Linear(in_features=hidden_size_shared, out_features=hidden_size_shared))
        self.shared_MLP = nn.Sequential(*self.shared_MLP)

        self.features_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=features_size, out_features=features_size), 
            act_function, 
            nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=features_size, out_features=features_size),
        )

        self.final_MLP = []
        for _ in range(n_layers_mlp):
            self.final_MLP.extend([torch.nn.Linear(in_features=hidden_size_shared + features_size, out_features=hidden_size_shared + features_size), act_function, nn.Dropout(p=dropout)])
        self.final_MLP.append(torch.nn.Linear(in_features=hidden_size_shared + features_size, out_features=1))
        self.final_MLP = nn.Sequential(*self.final_MLP)
        self.double()


    def forward(self, sequences: rnn.PackedSequence, features: torch.Tensor):
        # encode the sequences
        encoded_sequences = self.encoding_MLP(sequences.data)
        encoded_sequences = rnn.PackedSequence(data=encoded_sequences, batch_sizes=sequences.batch_sizes)
        # run the recurrent block
        hidden_states = self.recurrent_block(encoded_sequences)[0]
        # vectorize the porcess of all the hidden states of the batch
        processed_hidden_states = self.shared_MLP(hidden_states.data) 
        # create a new PackedSeequence class and unpack it
        processed_hidden_states = rnn.PackedSequence(data=processed_hidden_states, batch_sizes=hidden_states.batch_sizes) 
        unpacked_hidden_state, _ = rnn.pad_packed_sequence(processed_hidden_states, batch_first=True)
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
        pooled_hidden_states = torch.cat([pool_function(h) for h in unpacked_hidden_state], dim=0)
        processed_features = self.features_mlp(features)
        # generate the final prediction
        return self.final_MLP(torch.cat([pooled_hidden_states, processed_features], dim=1)).flatten()
    



class OligoLSTM(OligoRNN):


    def __init__(
            self, 
            input_size: int, 
            features_size: int, 
            hidden_size: int, 
            hidden_size_ecoder: int, 
            n_layers: int, 
            pool: str = 'max', 
            act_function: str = "relu", 
            n_layers_mlp: int = 1, 
            n_layers_encoder: int = 1, 
            dropout: float = 0, 
            bidirectional: bool = False
        ) -> None:

        super().__init__(
            input_size=input_size, 
            features_size=features_size, 
            hidden_size=hidden_size, 
            hidden_size_ecoder=hidden_size_ecoder, 
            n_layers=n_layers, 
            pool=pool, 
            act_function=act_function,  
            n_layers_mlp=n_layers_mlp,
            n_layers_encoder=n_layers_encoder,
            dropout=dropout, 
            bidirectional=bidirectional
        )
        self.recurrent_block = torch.nn.LSTM(input_size=hidden_size_ecoder, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)
        self.double()