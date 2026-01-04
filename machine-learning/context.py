import math
import torch
import torch.nn as nn

# ---- Logarithmus ----

x = 10

print(math.log(x)) 

# in python ist der log der natürliche logaritmus mit der euler'schen Zahl e definiert.
# also wie hoch muss ich e potenzieren um x zu erhalten.
# die ausgabe in prinzip die höhe des exponenten.
# e ist die basis des logaritmus.

# ---- Exponentialfunktion ----

x = 10

print(math.exp(x))
# in python ist die exp die euler'sche Zahl e hoch x.
# wenn wir x = 10 setzen, dann ist die ausgabe die euler'sche Zahl e hoch 10.
# e ist die basis der exponentialfunktion.

class PositionalEncoding(nn.Module):
    """Positional Encoding für Sequenzen"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model) #erstellt ein Tensor mit max_len Zeilen und d_model Spalten, initialisiert mit 0
        position = torch.arange(0, max_len).unsqueeze(1).float() #erstellt ein Tensor mit max_len Zeilen und 1 Spalte, initialisiert mit 0
        # unsqueeze ist notwendig, da man einen 1D-Tensor nicht mit einem 2D-Tensor multiplizieren kann.
        # position wird zu einem 2D-tensor mit dem shape 100x1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
