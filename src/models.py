import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange

'''
class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.Dropout(0.5),
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.Dropout(0.5),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.5,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        print(f"ConvBlock: {in_dim} -> {out_dim}")

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")

        self.pooling = nn.MaxPool1d(2)
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm2 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout1 = nn.Dropout(p_drop)
        self.dropout2 = nn.Dropout(p_drop)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.dropout1(X)

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))



        return self.dropout2(X)

'''

class Patching(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.net = Rearrange("b c (s ps)  -> b s (ps c)", ps=patch_size)
    
    def forward(self, x):
        x = self.net(x)
        return x


class LinearProjection(nn.Module):
    def __init__(self, patch_dim, dim):
        super().__init__()
        self.net = nn.Linear(patch_dim, dim)
    
    def forward(self, x):
        x = self.net(x)
        return x


class Embedding(nn.Module):
    def __init__(self, dim, n_patches):
        super().__init__()
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.pos_embedding = nn.Parameter(torch.randn(n_patches + 1, dim))
    
    def forward(self, x):
        batch_size, _, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        x = torch.concat([cls_tokens, x], dim=1)

        x += self.pos_embedding

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim_heads = dim // n_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.split_into_heads = Rearrange("b n (h d) -> b h n d", h=self.n_heads)

        self.softmax  = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.3)

        self.concat = Rearrange("b h n d -> b n (h d)", h = self.n_heads)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        logit = torch.matmul(q, k.transpose(-1, -2)) / (self.dim_heads ** 0.5)
        attention_weight = self.softmax(logit)
        attention_weight = self.dropout(attention_weight)

        out = torch.matmul(attention_weight, v)
        out = self.concat(out)

        return out


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class TransfomerEncoder(nn.Module):
    def __init__(self,dim,n_heads, mlp_dim, depth):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.multi_head_attention = MultiHeadAttention(dim=dim, n_heads=n_heads)
        self.mlp = MLP(dim=dim, hidden_dim=mlp_dim)
        self.depth = depth
    
    def forward(self, x):
        for _ in range(self.depth):
            x = self.multi_head_attention(self.norm(x)) + x
            x = self.mlp(self.norm(x)) + x

        return x
    

class MLPHead(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim),
            #nn.Dropout(0.1)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x


class TransFormer(nn.Module):
    def __init__(self, num_classes, seq_len, channels, patch_size=16, dim=128, depth=4, n_heads=1, hidden_dim=128):
        super().__init__()

        n_pathes = seq_len // patch_size
        patch_dim = channels * patch_size
        self.depth = depth

        self.patching = Patching(patch_size=patch_size)
        self.linear_projection = LinearProjection(patch_dim=patch_dim, dim=dim)
        self.embedding = Embedding(dim=dim, n_patches=n_pathes)
        self.transformer_encoder = TransfomerEncoder(dim=dim, n_heads=n_heads, mlp_dim=hidden_dim, depth=depth)
        self.mlp_head = MLPHead(dim=dim, out_dim=num_classes)

    def forward(self, x):
        x = self.patching(x)
        x = self.linear_projection(x)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        x = self.mlp_head(x)

        return x
