import logging
import multiprocessing as mp
from sys import argv
import torch
import dataclasses
from typing import Tuple, Union, List, Callable, Optional
from torch import nn
import torchaudio
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, hidden_size: int):
        super().__init__()

        self.energy = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, input):
        energy = self.energy(input)
        alpha = torch.softmax(energy, dim=-2)
        return (input * alpha).sum(dim=-2)


class CRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=config.cnn_out_channels,
                kernel_size=config.kernel_size, stride=config.stride
            ),
            nn.Flatten(start_dim=1, end_dim=2),
        )

        self.conv_out_frequency = (config.n_mels - config.kernel_size[0]) // \
            config.stride[0] + 1
        
        self.gru = nn.GRU(
            input_size=self.conv_out_frequency * config.cnn_out_channels,
            hidden_size=config.hidden_size,
            num_layers=config.gru_num_layers,
            bidirectional=config.bidirectional,
            batch_first=True
        )

        self.attention = Attention(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
    
    def forward(self, input):
        input = input.unsqueeze(dim=1)
        conv_output = self.conv(input).transpose(-1, -2)
        gru_output, _ = self.gru(conv_output)
        contex_vector = self.attention(gru_output)
        output = self.classifier(contex_vector)
        return output


class LogMelspec:
    def __init__(self, config):
        
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=config.n_mels
        ).to(config.device)

    def __call__(self, batch):
        return torch.log(self.melspec(batch).clamp_(min=1e-9, max=1e9))


class StreamCRNN(nn.Module):
    def __init__(self, config, max_window_length=100, streaming_step_size=5):
        super().__init__()
        self.config = config

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=config.cnn_out_channels,
                kernel_size=config.kernel_size, stride=config.stride
            ),
            nn.Flatten(start_dim=1, end_dim=2),
        )

        self.conv_out_frequency = (config.n_mels - config.kernel_size[0]) // \
            config.stride[0] + 1
        
        self.gru = nn.GRU(
            input_size=self.conv_out_frequency * config.cnn_out_channels,
            hidden_size=config.hidden_size,
            num_layers=config.gru_num_layers,
            dropout=0.1,
            bidirectional=config.bidirectional,
            batch_first=True
        )

        self.attention = Attention(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

        self.max_window_length = max_window_length
        self.streaming_step_size = streaming_step_size

        self.melspec = LogMelspec(config)
    
    def forward(self, input):
        input = input.unsqueeze(dim=1)
        conv_output = self.conv(input).transpose(-1, -2)
        gru_output, _ = self.gru(conv_output)
        contex_vector = self.attention(gru_output)
        output = self.classifier(contex_vector)
        return output


    def stream_forward(self, input, hidden=None):
        input = input.unsqueeze(dim=1)
        conv_output = self.conv(input).transpose(-1, -2)
        gru_output, hidden = self.gru(conv_output, hidden) if hidden is not None else self.gru(conv_output)
        contex_vector = self.attention(gru_output)
        output = self.classifier(contex_vector)
        return output, hidden

    def stream(self, input):
        self.eval()
        input = input.sum(dim=0)
        input = self.melspec(input.unsqueeze(0).to(self.config.device))
        kernel_width = self.conv[0].kernel_size[1]
        stride = self.conv[0].stride[-1]
        max_window_length = self.max_window_length - (self.max_window_length - kernel_width) % stride
        hidden = None
        
        output, hidden = self.stream_forward(input[:,:,:max_window_length])
        result = output.unsqueeze(1)

        last_i = None
        for i in range(max_window_length, input.shape[-1], self.streaming_step_size):
            output, hidden = model.stream_forward(input[:,:, i - max_window_length + 1 : i + 1], hidden=hidden)
            output = output.unsqueeze(1)
            prob = F.softmax(output, dim=-1).detach().numpy().flatten()[1]
            if prob > 0.9 and (i - max_window_length // 4) // 100 != last_i:
                print('key word detected at', (i - max_window_length // 4) // 100, 'second')
                last_i = (i - max_window_length // 4) // 100
        


@dataclasses.dataclass
class StudentConfig:
    keyword: str = 'sheila'  # We will use 1 key word -- 'sheila'
    batch_size: int = 128
    learning_rate: float = 1e-3 # 3e-4
    weight_decay: float = 1e-5 # 1e-5
    num_epochs: int = 40 # 20
    n_mels: int = 40
    cnn_out_channels: int = 2 # 8
    kernel_size: Tuple[int, int] = (5, 20) # (5, 20)
    stride: Tuple[int, int] = (2, 8) # (2, 8)
    hidden_size: int = 22 # 64
    gru_num_layers: int = 1 # 2
    bidirectional: bool = False
    num_classes: int = 2
    sample_rate: int = 16000
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    T: float = 10
    a: float = 0.6


if __name__ == "__main__":
    sconfig = StudentConfig()
    model = torch.load("kws.pth", map_location=torch.device('cpu')).eval()
    torch.save(model.state_dict(), 'kws.pt')

    model = StreamCRNN(sconfig, max_window_length = 100, streaming_step_size=5)
    model.load_state_dict(torch.load('kws.pt', map_location='cpu'))
    model.eval()
    audio_name = 'audio.wav'
    if len(argv) == 2:
        audio_name = argv[1]
    elif len(argv) > 2:
        print(f'Usage: {argv[0]} path_to_wav')
        exit(0)
    wav, sr = torchaudio.load(audio_name)

    model.stream(wav)
    
            
