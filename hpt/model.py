"""Log-mel front end + CRNN acoustic models with onset<-velocity and frame<-onset/offset conditioning (paper Sec. III-A, Fig. 3).
Attribute names follow the official checkpoint so it loads with strict=True (see load_official)."""
import math
import librosa, torch, torch.nn as nn, torch.nn.functional as F

SR, N_FFT, HOP, N_MELS, FMIN, FMAX = 16000, 2048, 160, 229, 30, 8000
FPS = SR // HOP  # 100 frames per second
N_KEYS, BEGIN_NOTE = 88, 21  # piano keys A0..C8 = MIDI 21..108
BN_MOMENTUM = 0.01


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)


def init_gru(rnn):
    """Reference init (forward direction only): gates uniform(+-sqrt(3/fan_in)), hidden 'new' gate orthogonal."""
    def uniform(w):
        bound = math.sqrt(3 / w.shape[1])
        nn.init.uniform_(w, -bound, bound)
    for i in range(rnn.num_layers):
        for name, inits in (("weight_ih_l", [uniform] * 3), ("weight_hh_l", [uniform, uniform, nn.init.orthogonal_])):
            w = getattr(rnn, f"{name}{i}")
            gate = w.shape[0] // 3
            for g, f in enumerate(inits):
                f(w[g * gate:(g + 1) * gate])
        nn.init.zeros_(getattr(rnn, f"bias_ih_l{i}"))
        nn.init.zeros_(getattr(rnn, f"bias_hh_l{i}"))


class LogMel(nn.Module):
    """|STFT|^2 -> Slaney mel (librosa defaults) -> 10*log10(max(x, 1e-10)); equals torchlibrosa with ref=1, top_db=None."""
    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(N_FFT), persistent=False)
        mel = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX).T
        self.register_buffer("melW", torch.tensor(mel, dtype=torch.float32), persistent=False)

    def forward(self, wav):  # (B, samples) -> (B, 1, T, N_MELS) with T = samples // HOP + 1
        spec = torch.stft(wav, N_FFT, HOP, window=self.window, center=True, pad_mode="reflect", return_complex=True)
        mel = (spec.abs() ** 2).transpose(1, 2) @ self.melW
        return (10 * torch.log10(mel.clamp(min=1e-10))).unsqueeze(1)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout, BN_MOMENTUM)
        self.bn2 = nn.BatchNorm2d(cout, BN_MOMENTUM)
        init_layer(self.conv1)
        init_layer(self.conv2)

    def forward(self, x):  # pool the frequency axis by 2, keep the time resolution
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, (1, 2))


class AcousticModel(nn.Module):
    """4 conv blocks (48/64/96/128 ch) -> FC 768 -> 2-layer biGRU(256) -> FC sigmoid."""
    def __init__(self, classes_num):
        super().__init__()
        self.conv_block1 = ConvBlock(1, 48)
        self.conv_block2 = ConvBlock(48, 64)
        self.conv_block3 = ConvBlock(64, 96)
        self.conv_block4 = ConvBlock(96, 128)
        self.fc5 = nn.Linear(128 * (N_MELS // 16), 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, BN_MOMENTUM)
        self.gru = nn.GRU(768, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, classes_num)
        init_layer(self.fc5)
        init_gru(self.gru)
        init_layer(self.fc)

    def forward(self, x):  # (B, 1, T, mel) -> (B, T, classes)
        for block in (self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4):
            x = F.dropout(block(x), 0.2, self.training)
        x = x.transpose(1, 2).flatten(2)  # (B, T, channels * mel)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, 0.5, self.training)
        x, _ = self.gru(x)
        x = F.dropout(x, 0.5, self.training)
        return torch.sigmoid(self.fc(x))


class NoteModel(nn.Module):
    """Regress_onset_offset_frame_velocity_CRNN in the reference."""
    def __init__(self):
        super().__init__()
        self.logmel = LogMel()
        self.bn0 = nn.BatchNorm2d(N_MELS, BN_MOMENTUM)
        self.frame_model = AcousticModel(N_KEYS)
        self.reg_onset_model = AcousticModel(N_KEYS)
        self.reg_offset_model = AcousticModel(N_KEYS)
        self.velocity_model = AcousticModel(N_KEYS)
        self.reg_onset_gru = nn.GRU(N_KEYS * 2, 256, batch_first=True, bidirectional=True)
        self.reg_onset_fc = nn.Linear(512, N_KEYS)
        self.frame_gru = nn.GRU(N_KEYS * 3, 256, batch_first=True, bidirectional=True)
        self.frame_fc = nn.Linear(512, N_KEYS)
        init_gru(self.reg_onset_gru)
        init_gru(self.frame_gru)
        init_layer(self.reg_onset_fc)
        init_layer(self.frame_fc)

    def forward(self, wav):
        x = self.bn0(self.logmel(wav).transpose(1, 3)).transpose(1, 3)  # BatchNorm over mel bins
        frame = self.frame_model(x)
        onset = self.reg_onset_model(x)
        offset = self.reg_offset_model(x)
        velocity = self.velocity_model(x)
        # velocity conditions the onset regression
        x, _ = self.reg_onset_gru(torch.cat((onset, onset ** 0.5 * velocity.detach()), dim=2))
        onset = torch.sigmoid(self.reg_onset_fc(F.dropout(x, 0.5, self.training)))
        # onsets and offsets condition the frame-wise output
        x, _ = self.frame_gru(torch.cat((frame, onset.detach(), offset.detach()), dim=2))
        frame = torch.sigmoid(self.frame_fc(F.dropout(x, 0.5, self.training)))
        return {"reg_onset_output": onset, "reg_offset_output": offset, "frame_output": frame, "velocity_output": velocity}


class PedalModel(nn.Module):
    """Regress_pedal_CRNN: three single-output acoustic models (pedal onset/offset regression, pedal frame)."""
    def __init__(self):
        super().__init__()
        self.logmel = LogMel()
        self.bn0 = nn.BatchNorm2d(N_MELS, BN_MOMENTUM)
        self.reg_pedal_onset_model = AcousticModel(1)
        self.reg_pedal_offset_model = AcousticModel(1)
        self.reg_pedal_frame_model = AcousticModel(1)

    def forward(self, wav):
        x = self.bn0(self.logmel(wav).transpose(1, 3)).transpose(1, 3)
        return {"reg_pedal_onset_output": self.reg_pedal_onset_model(x),
                "reg_pedal_offset_output": self.reg_pedal_offset_model(x),
                "pedal_frame_output": self.reg_pedal_frame_model(x)}


class NotePedalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.note_model, self.pedal_model = NoteModel(), PedalModel()

    def forward(self, wav):
        return {**self.note_model(wav), **self.pedal_model(wav)}


def load_official(model, path):
    """Load the authors' released note+pedal checkpoint; strict except the STFT/mel buffers we compute ourselves."""
    ck = torch.load(path, map_location="cpu", weights_only=True)["model"]
    for part in ("note_model", "pedal_model"):
        getattr(model, part).load_state_dict({k: v for k, v in ck[part].items() if "extractor" not in k}, strict=True)
    return model
