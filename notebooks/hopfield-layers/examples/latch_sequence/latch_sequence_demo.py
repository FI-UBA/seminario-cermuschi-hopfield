#!/usr/bin/env python
# coding: utf-8

# <h1>Example: Latch Problem</h1>
# 
# We study an easy example of learning long-term dependencies by using a simple <i>latch task</i> (see [Hochreiter and Mozer](https://link.springer.com/chapter/10.1007/3-540-44668-0_92)). The essence of this task is that a sequence of inputs is presented, beginning with one of two symbols, <b>A</b> or <b>B</b>, and after a variable number of time steps, the model has to output a corresponding symbol. Thus, the task requires memorizing the original input over time. It has to be noted, that both class-defining symbols must only appear at the first position of a sequence. This task was specifically designed to demonstrate the capability of recurrent neural networks to capture long term dependencies. This demonstration shows, that <code>Hopfield</code>, <code>HopfieldPooling</code> and <code>HopfieldLayer</code> adapt extremely fast to this specific task, concentrating only on the first entry of the sequence.
# 
# This demonstration instructs how to apply <code>Hopfield</code>, <code>HopfieldPooling</code> and <code>HopfieldLayer</code> for an exemplary sequential task, potentially substituting LSTM and GRU layers.
# 
# NOTA BENE: No tweeking of the exemplary LSTM network is done. The focus is put on the technical details. Feel free to tune yourself and see what works better :)
# 
# <h3 style="color:rgb(208,90,80)">In the chapters <a href="#Adapt-Hopfield-based-Network">Adapt Hopfield-based Network</a>, <a href="#Adapt-Hopfield-based-Pooling">Adapt Hopfield-based Pooling</a> and <a href="#Adapt-Hopfield-based-Lookup">Adapt Hopfield-based Lookup</a> you can explore and try the new functionalities of our new Hopfield layer.</h3>
# 
# In order to run this notebook, a few modules need to be imported. The installation of third-party modules is <i>not</i> covered here.

# In[1]:


# Import general modules used e.g. for plotting.
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
import torch

# Importing Hopfield-specific modules.
from hflayers import Hopfield, HopfieldPooling, HopfieldLayer
from hflayers.auxiliary.data import LatchSequenceSet

# Import auxiliary modules.
from distutils.version import LooseVersion
from typing import List, Tuple

# Importing PyTorch specific modules.
from torch import Tensor
from torch.nn import Flatten, Linear, LSTM, Module, Sequential
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Set plotting style.
sns.set()


# Specific minimum versions of Python itself as well as of some used modules is required.

# In[2]:


python_check = '(\u2713)' if sys.version_info >= (3, 8) else '(\u2717)'
pytorch_check = '(\u2713)' if torch.__version__ >= LooseVersion(r'1.5') else '(\u2717)'

print(f'Installed Python version:  {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} {python_check}')
print(f'Installed PyTorch version: {torch.__version__} {pytorch_check}')


# <h3 style="color:rgb(0,120,170)">Create Dataset</h3>
# 
# We study an easy example of learning long-term dependencies by using a simple <i>latch task</i>. 
# The latch task was introcuded by Hochreiter and Mozer:<br>
# <cite>Sepp Hochreiter, Michael Mozer, 2001. A discrete probabilistic memory model for discovering dependencies in time. Artificial Neural Networks -- ICANN 2001, 13, pp.661-668.</cite><br><br>
# The essence of this task is that a sequence of inputs is presented, beginning with one of two symbols, <b>A</b> or <b>B</b>, and after a variable number of time steps, the model has to output a corresponding symbol. Thus, the task requires memorizing the original input over time. It has to be noted, that both class-defining symbols must only appear at the first position of an instance. Defining arguments are:
# <br><br>
# <table>
#     <tr>
#         <th>Argument</th>
#         <th>Value (used in this demo)</th>
#         <th>Description</th>
#     </tr>
#     <tr>
#         <th><code>num_samples</code></th>
#         <th>4096</th>
#         <th>Amount of samples of the full dataset.</th>
#     </tr>
#     <tr>
#         <th><code>num_instances</code></th>
#         <th>32</th>
#         <th>Amount of instances per sample (sample length).</th>
#     </tr>
#     <tr>
#         <th><code>num_characters</code></th>
#         <th>20</th>
#         <th>Amount of different characters (size of the one-hot encoded vector).</th>
#     </tr>
# </table>
# 
# Let's define the dataset using previously mentioned properties as well as a logging directory for storing all auxiliary outputs like performance plots.

# In[3]:


latch_sequence_set = LatchSequenceSet(
    num_samples=4096,
    num_instances=32,
    num_characters=20)


# In[ ]:


it = iter(latch_sequence_set)


# In[8]:


next(it)


# In[9]:


log_dir = f'resources/'
os.makedirs(log_dir, exist_ok=True)


# <h3 style="color:rgb(0,120,170)">Create Auxiliaries</h3>
# 
# Before digging into Hopfield-based networks, a few auxiliary variables and functions need to be defined. This is nothing special with respect to Hopfield-based networks, but rather common preparation work of (almost) every machine learning setting (e.g. definition of a <i>data loader</i> as well as a <i>training loop</i>). We will see, that this comprises the most work of this whole demo.

# In[10]:


device = torch.device(r'cuda:0' if torch.cuda.is_available() else r'cpu')

# Create data loader of training set.
sampler_train = SubsetRandomSampler(list(range(512, 4096 - 512)))
data_loader_train = DataLoader(dataset=latch_sequence_set, batch_size=32, sampler=sampler_train)

# Create data loader of validation set.
sampler_eval = SubsetRandomSampler(list(range(512)) + list(range(4096 - 512, 4096)))
data_loader_eval = DataLoader(dataset=latch_sequence_set, batch_size=32, sampler=sampler_eval)


# In[11]:


def train_epoch(network: Module,
                optimiser: AdamW,
                data_loader: DataLoader
               ) -> Tuple[float, float]:
    """
    Execute one training epoch.
    
    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader: data loader instance providing training data
    :return: tuple comprising training loss as well as accuracy
    """
    network.train()
    losses, accuracies = [], []
    for sample_data in data_loader:
        data, target = sample_data[r'data'], sample_data[r'target']
        data, target = data.to(device=device), target.to(device=device)

        # Process data by Hopfield-based network.
        model_output = network.forward(input=data)

        # Update network parameters.
        optimiser.zero_grad()
        loss = binary_cross_entropy_with_logits(input=model_output, target=target, reduction=r'mean')
        loss.backward()
        clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimiser.step()

        # Compute performance measures of current model.
        accuracy = (model_output.sigmoid().round() == target).to(dtype=torch.float32).mean()
        accuracies.append(accuracy.detach().item())
        losses.append(loss.detach().item())
    
    # Report progress of training procedure.
    return (sum(losses) / len(losses), sum(accuracies) / len(accuracies))


def eval_iter(network: Module,
              data_loader: DataLoader
             ) -> Tuple[float, float]:
    """
    Evaluate the current model.
    
    :param network: network instance to evaluate
    :param data_loader: data loader instance providing validation data
    :return: tuple comprising validation loss as well as accuracy
    """
    network.eval()
    with torch.no_grad():
        losses, accuracies = [], []
        for sample_data in data_loader:
            data, target = sample_data[r'data'], sample_data[r'target']
            data, target = data.to(device=device), target.to(device=device)

            # Process data by Hopfield-based network.
            model_output = network.forward(input=data)
            loss = binary_cross_entropy_with_logits(input=model_output, target=target, reduction=r'mean')

            # Compute performance measures of current model.
            accuracy = (model_output.sigmoid().round() == target).to(dtype=torch.float32).mean()
            accuracies.append(accuracy.detach().item())
            losses.append(loss.detach().item())

        # Report progress of validation procedure.
        return (sum(losses) / len(losses), sum(accuracies) / len(accuracies))


def operate(network: Module,
            optimiser: AdamW,
            data_loader_train: DataLoader,
            data_loader_eval: DataLoader,
            num_epochs: int = 1
           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train the specified network by gradient descent using backpropagation.
    
    :param network: network instance to train
    :param optimiser: optimiser instance responsible for updating network parameters
    :param data_loader_train: data loader instance providing training data
    :param data_loader_eval: data loader instance providing validation data
    :param num_epochs: amount of epochs to train
    :return: data frame comprising training as well as evaluation performance
    """
    losses, accuracies = {r'train': [], r'eval': []}, {r'train': [], r'eval': []}
    for epoch in range(num_epochs):
        
        # Train network.
        performance = train_epoch(network, optimiser, data_loader_train)
        losses[r'train'].append(performance[0])
        accuracies[r'train'].append(performance[1])
        
        # Evaluate current model.
        performance = eval_iter(network, data_loader_eval)
        losses[r'eval'].append(performance[0])
        accuracies[r'eval'].append(performance[1])
    
    # Report progress of training and validation procedures.
    return pd.DataFrame(losses), pd.DataFrame(accuracies)


# In[12]:


def set_seed(seed: int = 42) -> None:
    """
    Set seed for all underlying (pseudo) random number sources.
    
    :param seed: seed to be used
    :return: None
    """
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_performance(loss: pd.DataFrame,
                     accuracy: pd.DataFrame,
                     log_file: str
                    ) -> None:
    """
    Plot and save loss and accuracy.
    
    :param loss: loss to be plotted
    :param accuracy: accuracy to be plotted
    :param log_file: target file for storing the resulting plot
    :return: None
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    
    loss_plot = sns.lineplot(data=loss, ax=ax[0])
    loss_plot.set(xlabel=r'Epoch', ylabel=r'Cross-entropy Loss')
    
    accuracy_plot = sns.lineplot(data=accuracy, ax=ax[1])
    accuracy_plot.set(xlabel=r'Epoch', ylabel=r'Accuracy')
    
    ax[1].yaxis.set_label_position(r'right')
    fig.tight_layout()
    fig.savefig(log_file)
    plt.show(fig)


# <h2>LSTM-based Network</h2>
# 
# The instantiation of the heart of an LSTM-based network, the module <code>LSTM</code>, is rather straightforward. Only <i>two</i> arguments, the size of the input as well as the site of the hidden state, need to be set.
# <br><br>
# <table>
#     <tr>
#         <th>Argument</th>
#         <th>Value (used in this demo)</th>
#         <th>Description</th>
#     </tr>
#     <tr>
#         <th><code>input_size</code></th>
#         <th>num_characters (20)</th>
#         <th>Size (depth) of the input.</th>
#     </tr>
#     <tr>
#         <th><code>hidden_size</code></th>
#         <th>4</th>
#         <th>Size (depth) of the hidden state.</th>
#     </tr>
#     <tr>
#         <th><code>...</code></th>
#         <th>default</th>
#         <th>The remaining arguments are not explicitly used in this example.</th>
#     </tr>
# </table>
# 
# An additional output projection is defined, to downproject the hidden state of the last time step of the <code>LSTM</code> to the correct output size. Afterwards, everything is wrapped into a container of type <code>torch.nn.Sequential</code> and a corresponding optimiser is defined.

# In[13]:


class LSTMNetwork(Module):
    def __init__(self, input_size: int, hidden_size: int):
        """
        Initialize a new instance of an LSTM-based network.
        
        :param input size: size (depth) of the input
        :param hidden_size: size (depth) of the hidden state
        """
        super(LSTMNetwork, self).__init__()
        self.lstm = LSTM(input_size, hidden_size, batch_first=True)
        self.projection = Linear(hidden_size, 1)
    
    def forward(self, input: Tensor) -> Tensor:
        """
        Compute result of LSTM-based network on specified data.
        
        :param input: data to be processed by the LSTM-based network
        :return: result as computed by the LSTM-based network
        """
        out, _ = self.lstm.forward(input=input)     
        return self.projection.forward(input=out[:, -1, :]).flatten()


# In[14]:


set_seed()
network = LSTMNetwork(
    input_size=latch_sequence_set.num_characters,
    hidden_size=4).to(device=device)
optimiser = AdamW(params=network.parameters(), lr=1e-3)


# <h3 style="color:rgb(0,120,170)">Operate LSTM-based Network</h3>

# In[15]:


losses, accuracies = operate(
    network=network,
    optimiser=optimiser,
    data_loader_train=data_loader_train,
    data_loader_eval=data_loader_eval,
    num_epochs=20)


# In[16]:


plot_performance(loss=losses, accuracy=accuracies, log_file=f'{log_dir}/lstm_base.pdf')


# <h2>Hopfield-based Network</h2>
# 
# The instantiation of the heart of a Hopfield-based network, the module <code>Hopfield</code>, is even simpler. Only <i>one</i> argument, the size of the input, needs to be set.
# <br><br>
# <table>
#     <tr>
#         <th>Argument</th>
#         <th>Value (used in this demo)</th>
#         <th>Description</th>
#     </tr>
#     <tr>
#         <th><code>input_size</code></th>
#         <th>num_characters (20)</th>
#         <th>Size (depth) of the input (state pattern).</th>
#     </tr>
#     <tr>
#         <th><code>...</code></th>
#         <th>default</th>
#         <th>The remaining arguments are not explicitly used in this example.</th>
#     </tr>
# </table>
# 
# An additional output projection is defined, to downproject the result of <code>Hopfield</code> to the correct output size. Afterwards, everything is wrapped into a container of type <code>torch.nn.Sequential</code> and a corresponding optimiser is defined. Now the Hopfield-based network and all auxiliaries are set up and ready to <i>associate</i>!

# In[17]:


set_seed()
hopfield = Hopfield(
    input_size=latch_sequence_set.num_characters)


# In[18]:


output_projection = Linear(in_features=hopfield.output_size * latch_sequence_set.num_instances, out_features=1)
network = Sequential(hopfield, Flatten(), output_projection, Flatten(start_dim=0)).to(device=device)
optimiser = AdamW(params=network.parameters(), lr=1e-3)


# <h3 style="color:rgb(0,120,170)">Operate Hopfield-based Network</h3>

# In[19]:


losses, accuracies = operate(
    network=network,
    optimiser=optimiser,
    data_loader_train=data_loader_train,
    data_loader_eval=data_loader_eval,
    num_epochs=20)


# In[20]:


plot_performance(loss=losses, accuracy=accuracies, log_file=f'{log_dir}/hopfield_base.pdf')


# <h3 style="color:rgb(0,120,170)">Adapt Hopfield-based Network</h3>
# <h3 style="color:rgb(208,90,80)">We can now explore the functionality of our Hopfield layer <code>Hopfield</code>.</h3>
# 
# As described in the paper the Hopfield layer allows:
# - association of two sets
# - multiple updates
# - variable beta
# - changing the dimension of the associative space
# - pattern normalization
# - static patterns for fixed pattern search
# 
# This time, additional arguments are set to influence the training as well as the validation performance of the Hopfield-based network.
# <br><br>
# <table>
#     <tr>
#         <th>Argument</th>
#         <th>Value (used in this demo)</th>
#         <th>Description</th>
#     </tr>
#     <tr>
#         <th><code>input_size</code></th>
#         <th>num_characters (20)</th>
#         <th>Size (depth) of the input (state pattern).</th>
#     </tr>
#     <tr>
#         <th><code>hidden_size</code></th>
#         <th>8</th>
#         <th>Size (depth) of the association space.</th>
#     </tr>
#     <tr>
#         <th><code>num_heads</code></th>
#         <th>8</th>
#         <th>Amount of parallel association heads.</th>
#     </tr>
#     <tr>
#         <th><code>update_steps_max</code></th>
#         <th>3</th>
#         <th>Number of updates in one Hopfield head.</th>
#     </tr>
#     <tr>
#         <th><code>scaling</code></th>
#         <th>0.25</th>
#         <th>Beta parameter that determines the kind of fixed point.</th>
#     </tr>
#     <tr>
#         <th><code>dropout</code></th>
#         <th>0.5</th>
#         <th>Dropout probability applied on the association matrix.</th>
#     </tr>
#     <tr>
#         <th><code>...</code></th>
#         <th>default</th>
#         <th>The remaining arguments are not explicitly used in this example.</th>
#     </tr>
# </table>

# In[ ]:


set_seed()
hopfield = Hopfield(
    input_size=latch_sequence_set.num_characters,
    hidden_size=8,
    num_heads=8,
    update_steps_max=3,
    scaling=0.25,
    dropout=0.5)


# In[ ]:


output_projection = Linear(in_features=hopfield.output_size * latch_sequence_set.num_instances, out_features=1)
network = Sequential(hopfield, Flatten(), output_projection, Flatten(start_dim=0)).to(device=device)
optimiser = AdamW(params=network.parameters(), lr=1e-3)


# In[ ]:


losses, accuracies = operate(
    network=network,
    optimiser=optimiser,
    data_loader_train=data_loader_train,
    data_loader_eval=data_loader_eval,
    num_epochs=20)


# In[ ]:


plot_performance(loss=losses, accuracy=accuracies, log_file=f'{log_dir}/hopfield_adapted.pdf')


# <h2>Hopfield-based Pooling</h2>
# 
# The previous examples manually downprojected the result of <code>Hopfield</code> by applying a linear layer afterwards. It would've also been possible to apply some kind of <i>pooling</i>. Exactly for <i>such</i> use cases, the module <code>HopfieldPooling</code> might be the right choice. Internally, a <i>state pattern</i> is trained, which in turn is used to compute pooling weights with respect to the input.

# In[21]:


set_seed()
hopfield_pooling = HopfieldPooling(
    input_size=latch_sequence_set.num_characters)


# In[22]:


output_projection = Linear(in_features=hopfield_pooling.output_size, out_features=1)
network = Sequential(hopfield_pooling, output_projection, Flatten(start_dim=0)).to(device=device)
optimiser = AdamW(params=network.parameters(), lr=1e-3)


# <h3 style="color:rgb(0,120,170)">Operate Hopfield-based Pooling</h3>

# In[23]:


losses, accuracies = operate(
    network=network,
    optimiser=optimiser,
    data_loader_train=data_loader_train,
    data_loader_eval=data_loader_eval,
    num_epochs=20)


# In[24]:


plot_performance(loss=losses, accuracy=accuracies, log_file=f'{log_dir}/hopfield_pooling.pdf')


# <h3 style="color:rgb(0,120,170)">Adapt Hopfield-based Pooling</h3>
# <h3 style="color:rgb(208,90,80)">We can now again explore the functionality of our Hopfield-based pooling layer <code>HopfieldPooling</code>.</h3>
# 
# Again, additional arguments are set to influence the training as well as the validation performance of the Hopfield-based pooling.
# <br><br>
# <table>
#     <tr>
#         <th>Argument</th>
#         <th>Value (used in this demo)</th>
#         <th>Description</th>
#     </tr>
#     <tr>
#         <th><code>input_size</code></th>
#         <th>num_characters (20)</th>
#         <th>Size (depth) of the input (state pattern).</th>
#     </tr>
#     <tr>
#         <th><code>hidden_size</code></th>
#         <th>8</th>
#         <th>Size (depth) of the association space.</th>
#     </tr>
#     <tr>
#         <th><code>num_heads</code></th>
#         <th>8</th>
#         <th>Amount of parallel association heads.</th>
#     </tr>
#     <tr>
#     <tr>
#         <th><code>update_steps_max</code></th>
#         <th>3</th>
#         <th>Number of updates in one Hopfield head.</th>
#     </tr>
#     <tr>
#         <th><code>scaling</code></th>
#         <th>0.25</th>
#         <th>Beta parameter that determines the kind of fixed point.</th>
#     </tr>
#     <tr>
#         <th><code>dropout</code></th>
#         <th>0.5</th>
#         <th>Dropout probability applied on the association matrix.</th>
#     </tr>
#     <tr>
#         <th><code>...</code></th>
#         <th>default</th>
#         <th>The remaining arguments are not explicitly used in this example.</th>
#     </tr>
# </table>

# In[ ]:


set_seed()
hopfield_pooling = HopfieldPooling(
    input_size=latch_sequence_set.num_characters,
    hidden_size=8,
    num_heads=8,
    update_steps_max=3,
    scaling=0.25,
    dropout=0.5)


# In[ ]:


output_projection = Linear(in_features=hopfield_pooling.output_size, out_features=1)
network = Sequential(hopfield_pooling, output_projection, Flatten(start_dim=0)).to(device=device)
optimiser = AdamW(params=network.parameters(), lr=1e-3)


# In[ ]:


losses, accuracies = operate(
    network=network,
    optimiser=optimiser,
    data_loader_train=data_loader_train,
    data_loader_eval=data_loader_eval,
    num_epochs=20)


# In[ ]:


plot_performance(loss=losses, accuracy=accuracies, log_file=f'{log_dir}/hopfield_pooling_adapted.pdf')


# <h2>Hopfield-based Lookup</h2>
# 
# In contrast to the first <code>Hopfield</code> setting, in which the <i>state patterns</i> as well as the <i>stored patterns</i> are directly dependent on the input, <code>HopfieldLayer</code> employs a trainable but fixed <i>stored pattern</i> matrix, which in turn acts as a learnable lookup table.

# In[ ]:


latch_samples_unique = [_[r'data'] for _ in data_loader_train]
latch_samples_unique = torch.cat(latch_samples_unique).view(-1, latch_samples_unique[0].shape[2]).unique(dim=0)


# In[ ]:


set_seed()
hopfield_lookup = HopfieldLayer(
    input_size=latch_sequence_set.num_characters,
    quantity=len(latch_samples_unique))


# In[ ]:


output_projection = Linear(in_features=hopfield_lookup.output_size * latch_sequence_set.num_instances, out_features=1)
network = Sequential(hopfield_lookup, Flatten(start_dim=1), output_projection, Flatten(start_dim=0)).to(device=device)
optimiser = AdamW(params=network.parameters(), lr=1e-3)


# <h3 style="color:rgb(0,120,170)">Operate Hopfield-based Lookup</h3>

# In[ ]:


losses, accuracies = operate(
    network=network,
    optimiser=optimiser,
    data_loader_train=data_loader_train,
    data_loader_eval=data_loader_eval,
    num_epochs=18)


# In[ ]:


plot_performance(loss=losses, accuracy=accuracies, log_file=f'{log_dir}/hopfield_lookup.pdf')


# <h3 style="color:rgb(0,120,170)">Adapt Hopfield-based Lookup</h3>
# <h3 style="color:rgb(208,90,80)">We can now again explore the functionality of our Hopfield-based lookup layer <code>HopfieldLayer</code>.</h3>
# 
# This <i>lookup setting</i> is especially pronounced, if the <i>state patterns</i> are initialized with a subset of the training set (and optionally provide the corresponding training targets as <i>pattern projection</i> inputs).
# 
# Again, additional arguments are set to increase the training as well as the validation performance of the Hopfield-based lookup.
# <br><br>
# <table>
#     <tr>
#         <th>Argument</th>
#         <th>Value (used in this demo)</th>
#         <th>Description</th>
#     </tr>
#     <tr>
#         <th><code>lookup_weights_as_separated</code></th>
#         <th>True</th>
#         <th>Separate lookup weights from lookup target weights (e.g. to set lookup target weights separately).</th>
#     </tr>
#     <tr>
#         <th><code>lookup_targets_as_trainable</code></th>
#         <th>False</th>
#         <th>Employ trainable lookup target weights (used as pattern projection input).</th>
#     </tr>
# </table>

# In[ ]:


set_seed()
hopfield_lookup = HopfieldLayer(
    input_size=latch_sequence_set.num_characters,
    quantity=len(latch_samples_unique),
    lookup_weights_as_separated=True,
    lookup_targets_as_trainable=False)


# Initialize the trainable but fixed <i>stored patterns</i> with all unique samples from the training set. In this way, the Hopfield-based lookup already starts with <i>meaningful</i> stored patterns (instead of random noise). This may enhance the performance of the network, especially at the beginning of the training.

# In[ ]:


with torch.no_grad():
    hopfield_lookup.lookup_weights[:] = latch_samples_unique.unsqueeze(dim=0)


# In[ ]:


output_projection = Linear(in_features=hopfield_lookup.output_size * latch_sequence_set.num_instances, out_features=1)
network = Sequential(hopfield_lookup, Flatten(start_dim=1), output_projection, Flatten(start_dim=0)).to(device=device)
optimiser = AdamW(params=network.parameters(), lr=1e-3)


# In[ ]:


losses, accuracies = operate(
    network=network,
    optimiser=optimiser,
    data_loader_train=data_loader_train,
    data_loader_eval=data_loader_eval,
    num_epochs=18)


# In[ ]:


plot_performance(loss=losses, accuracy=accuracies, log_file=f'{log_dir}/hopfield_lookup_adapted.pdf')

