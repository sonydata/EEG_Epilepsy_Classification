import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt



def extract_upper_triangle(corr_matrices):
    """
    Extract upper triangles from correlation matrices
    
    Args:
        corr_matrices: numpy array of shape (n_segments, n_channels, n_channels)
        
    Returns:
        numpy array of shape (n_segments, n_features) where n_features = n_channels*(n_channels-1)/2
    """
    n_segments, n_channels, _ = corr_matrices.shape
    n_features = n_channels * (n_channels - 1) // 2
    
    flattened = np.zeros((n_segments, n_features))
    
    for i in range(n_segments):
        # Get upper triangle indices (excluding diagonal)
        upper_indices = np.triu_indices(n_channels, k=1)
        # Extract values
        flattened[i] = corr_matrices[i][upper_indices]
    
    return flattened

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Final projection after concatenating heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Project Q, K, V
        Q = self.q_proj(query)  # (batch_size, seq_len, embed_dim)
        K = self.k_proj(key)    # (batch_size, seq_len, embed_dim)
        V = self.v_proj(value)  # (batch_size, seq_len, embed_dim)
        
        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # Apply softmax to get attention weights
        attn_weights = self.softmax(scores)  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)
        
        # Calculate weighted output
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Recompose heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # (batch_size, seq_len, embed_dim)
        
        # Pass through final projection
        output = self.out_proj(attn_output)  # (batch_size, seq_len, embed_dim)
        
        return output, attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length=100):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # Add positional encoding to input
        # x: [batch_size, seq_len, embed_dim]
        return x + self.pe[:, :x.size(1)]

class TimeSeriesAttentionClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_classes=2, dropout=0.1):
        super(TimeSeriesAttentionClassifier, self).__init__()
        
        # Project flattened correlation features to embedding space
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        # Project to embedding space
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Self-attention (use x for query, key, and value)
        residual = x
        x, attention_weights = self.attention(x, x, x)
        x = self.layer_norm1(x + residual)
        
        # Feed-forward network with residual connection
        residual = x
        x = self.ffn(x)
        x = self.layer_norm2(x + residual)
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits, attention_weights

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')
    
    return train_losses, val_losses, val_accuracies

def visualize_attention(model, inputs, n_sample=1):
    """Visualize attention maps for a sample"""
    model.eval()
    with torch.no_grad():
        _, attention_weights = model(inputs[:n_sample])
    
    # Get the first sample's attention weights from the first head
    attn_map = attention_weights[0, 0].cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attn_map, cmap='viridis')
    plt.colorbar()
    plt.title('Attention Map (First Head)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    plt.show()

def main():
    # Simulating your data for demonstration
    n_samples = 100  # Number of samples
    n_segments = 12  # 12 segments of 5s each
    n_channels = 21  # 21 channels
    
    # Create sample correlation matrices (replace with your actual data)
    np.random.seed(42)
    
    # Generate correlation matrices for each sample and segment
    corr_matrices = []
    labels = []
    
    for i in range(n_samples):
        # Create random correlation matrices
        sample_matrices = np.random.uniform(-1, 1, (n_segments, n_channels, n_channels))
        
        # Make them symmetric with diagonal of 1
        for j in range(n_segments):
            sample_matrices[j] = (sample_matrices[j] + sample_matrices[j].T) / 2
            np.fill_diagonal(sample_matrices[j], 1.0)
        
        # Extract upper triangles
        flattened = extract_upper_triangle(sample_matrices)
        corr_matrices.append(flattened)
        
        # Generate random label (0 or 1) for demonstration
        labels.append(np.random.randint(0, 2))
    
    # Convert to PyTorch tensors
    X = torch.tensor(np.array(corr_matrices), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.long)
    
    # Split into train and validation sets (80/20 split)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model parameters
    input_dim = 210  # Size of flattened upper triangle (21*20/2)
    embed_dim = 128  # Embedding dimension
    num_heads = 4    # Number of attention heads
    
    print(next(iter(train_loader))[0].shape)

    model = TimeSeriesAttentionClassifier(input_dim, embed_dim, num_heads)
    print(model)
    print # parameters
    print("*" * 20)
    print("Model Parameters:")
    print("*" * 20)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)

    # # Train the model
    # train_losses, val_losses, val_accuracies = train_model(
    #     model, train_loader, val_loader, num_epochs=20
    # )
    
    # # Plot training curves
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    
    # plt.subplot(1, 2, 2)
    # plt.plot(val_accuracies)
    # plt.xlabel('Epoch')
    # plt.ylabel('Validation Accuracy (%)')
    # plt.tight_layout()
    # plt.show()
    
    # # Visualize attention for a sample
    # visualize_attention(model, X_val)

if __name__ == "__main__":
    main()