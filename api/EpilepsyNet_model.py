import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



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
    def __init__(self, embed_dim, num_heads, dropout=0.3):
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
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Project Q, K, V
        Q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        K = self.k_proj(x)    # (batch_size, seq_len, embed_dim)
        V = self.v_proj(x)  # (batch_size, seq_len, embed_dim)
        
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
    def __init__(self, input_dim, embed_dim, num_heads, num_classes=2, dropout=0.2):
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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # batch_size, seq_len, input_dim = x.shape
        
        # Project to embedding space
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Self-attention (use x for query, key, and value)
        residual = x
        x, attention_weights = self.attention(x)
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

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4, weight_decay=1e-5,  patience=10, scheduler_factor=0.5, min_lr=1e-6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Changed from CrossEntropyLoss to BCELoss for binary classification with sigmoid
    criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add L2 regularization through weight_decay parameter in Adam
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=scheduler_factor,
        patience=patience,
        verbose=True,
        min_lr=min_lr
    )

    train_losses = []
    val_losses = []
    val_accuracies = []
    
    
    # Track best model and early stopping
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0
    early_stop_patience = patience * 2  # Stop after 2x the scheduler patience
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Convert labels to float and reshape for BCE loss
            labels = labels.float().view(-1, 1)
            
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
                # Convert labels to float and reshape for BCE loss
                labels = labels.float().view(-1, 1)
                
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # For binary classification with sigmoid, prediction is 1 if output > 0.5
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        # Learning rate scheduler step based on validation loss
        scheduler.step(val_loss)
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']


        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses, val_accuracies

