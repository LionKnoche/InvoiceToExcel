"""
Einfaches Transformer-Modell
Demonstriert die grundlegende Transformer-Architektur
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional Encoding für Sequenzen"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model) #erstellt ein Tensor mit max_len Zeilen und d_model Spalten, initialisiert mit 0
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SimpleTransformer(nn.Module):
    """
    Einfacher Transformer mit Encoder-Layers
    
    Args:
        vocab_size: Größe des Vokabulars
        d_model: Dimension des Modells (z.B. 128)
        nhead: Anzahl der Attention-Heads
        num_layers: Anzahl der Transformer-Layers
        dim_feedforward: Dimension der Feed-Forward-Layer
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, 
                 num_layers=2, dim_feedforward=512):
        super().__init__()
        self.d_model = d_model
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output Layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x


# Beispiel-Nutzung
if __name__ == "__main__":
    # Parameter
    vocab_size = 1000  # Größe des Vokabulars
    d_model = 128      # Modell-Dimension
    batch_size = 2     # Batch-Größe
    seq_len = 10       # Sequenz-Länge
    
    # Modell erstellen
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=4,
        num_layers=2
    )
    
    # Beispiel-Input (Token-IDs)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward Pass
    output = model(input_ids)
    
    print(f"Input Shape: {input_ids.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Modell-Parameter: {sum(p.numel() for p in model.parameters()):,}")
    
    # Modell speichern
    torch.save(model.state_dict(), 'transformer_model.pth')
    print("\nModell gespeichert als 'transformer_model.pth'")

