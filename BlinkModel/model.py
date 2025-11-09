import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import profile
from thop import clever_format

# FeatureIntra模块：用于交互处理head_query和eye_query
class FeatureIntra(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(FeatureIntra, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(feature_dim, num_heads)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, query, memory):
        B, T, D = query.size()
        query = query.reshape(B*T, 1, D).transpose(0, 1)
        memory = memory.reshape(B*T, -1, D).transpose(0, 1)

        # 计算多头注意力
        attn_output, _ = self.attention(query, memory, memory)
        attn_output = attn_output.transpose(0, 1).reshape(B, T, D)

        # 残差连接和层归一化
        query = attn_output
        query = self.norm1(query)

        # 前馈网络
        ffn_output = self.ffn(query)
        query = query + ffn_output  # 残差连接
        query = self.norm2(query)

        return query

class TimeSeriesModel(nn.Module):
    def __init__(self, feature_dim, num_heads, win_size=None, stride=1, ff_dim=512):
        super(TimeSeriesModel, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.stride = stride
        self.ff_dim = ff_dim  # Feed-Forward Network的维度
        
        # 使用 PyTorch 提供的 MultiheadAttention 层
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, feature_dim)
        )
        
        # LayerNorm 层
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

    def create_attention_mask(self, T):
        # Initialize a mask with zeros
        mask = torch.zeros(T, T)
    
        # 计算每个位置的开始和结束的范围
        indices = torch.arange(T)  # [0, 1, 2, ..., T-1]
        
        # 计算左侧窗口的范围
        start_indices = indices.unsqueeze(1) - self.win_size * self.stride
        start_indices = torch.max(start_indices, torch.zeros_like(start_indices))  # 保证最小为0
        
        # 计算右侧窗口的范围
        end_indices = indices.unsqueeze(1) + (self.win_size + 1) * self.stride
        end_indices = torch.min(end_indices, torch.full_like(end_indices, T))  # 保证最大不超过 T
    
        # 使用广播计算 mask
        mask = (indices.unsqueeze(0) >= start_indices) & (indices.unsqueeze(0) < end_indices)
        return mask

    def forward(self, x):
        B, T, D = x.size()
        
        attn_mask = None
        # 创建 attention mask
        if self.win_size is not None:
          attn_mask = self.create_attention_mask(T).to(x.device)
        
        # 1. 多头注意力计算
        x = x.permute(1, 0, 2)  # 转换为 (T, B, D) 形式
        
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask)
        x = x + attn_output  # 残差连接
        x = self.norm1(x)  # 应用 LayerNorm
        
        # 2. 前馈网络（FFN）计算
        ffn_output = self.ffn(x)
        x = x + ffn_output  # 残差连接
        x = self.norm2(x)  # 应用 LayerNorm
        
        # 将输出转换回 (B, T, D)
        x = x.permute(1, 0, 2)

        return x

# 模型整体框架：BlinkTransformerDecoder
class BlinkTransformerDecoder(nn.Module):
    def __init__(self, feature_dim=256, map_size = 60, infer_len = 64,  num_heads=8, num_layers_encoder=3, num_layers_decoder=6, roi_feature_encoder = False):
        super(BlinkTransformerDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.roi_feature_encoder = roi_feature_encoder
        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.feature_dim)
        # Head query 和 Eye query 交互模块
        self.query = nn.Parameter(torch.randn(infer_len, self.feature_dim))
        self.collect_roi_feature =  FeatureIntra(self.feature_dim, self.num_heads)
        # TimeSeriesModel: 用于处理时间序列
        if self.roi_feature_encoder:
          self.spatial_position = nn.Parameter(torch.randn(map_size, self.feature_dim))
          self.time_position = nn.Parameter(torch.randn(infer_len, self.feature_dim))
          self.feature_intra = nn.ModuleList([FeatureIntra(self.feature_dim, self.num_heads) for i in range(self.num_layers_decoder)])
          self.time_series_model_spatial = nn.ModuleList([TimeSeriesModel(self.feature_dim, self.num_heads) for i in range(self.num_layers_encoder)])            
          self.time_series_model_time = nn.ModuleList([TimeSeriesModel(self.feature_dim, self.num_heads) for i in range(self.num_layers_encoder)])
        
        self.time_series_model_query = nn.ModuleList([TimeSeriesModel(self.feature_dim, self.num_heads) for i in range(self.num_layers_decoder)])
        
        self.score_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Dropout(0),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 2)
        )

    def forward(self, blink_features, head_query, eye_query, return_query=False):
        B, T, D = head_query.size()
      
        blink_features = blink_features.reshape(B, T, -1, D)
        HW = blink_features.size(2)

        if self.roi_feature_encoder:
            blink_features = blink_features.reshape(B*T, HW, D)
            blink_features = blink_features + self.spatial_position[None].repeat(B*T, 1, 1)  ##abla spatial
            
            blink_features = blink_features.reshape(B, T, -1, D).transpose(1, 2)
            blink_features = blink_features.reshape(B*HW, T, D)
            blink_features = blink_features + self.time_position[None].repeat(B*HW, 1, 1)   ##abla time
            
            blink_features = blink_features.reshape(B, HW, T, D).transpose(1, 2)
    
            for i in range(self.num_layers_encoder):
                blink_features = blink_features.reshape(B*T, HW, D)#BT HW D
                blink_features = self.time_series_model_spatial[i](blink_features)  ##abla spatial

                blink_features = blink_features.reshape(B, T, HW, D).transpose(1, 2)
                blink_features = blink_features.reshape(B*HW, T, D)
                blink_features = self.time_series_model_time[i](blink_features) ##abla time
                
                blink_features = blink_features.reshape(B, HW, T, D).transpose(1, 2) ### B T HW D
       
        query = self.query[None].repeat(B, 1, 1) + head_query + eye_query
        query = self.collect_roi_feature(query, blink_features)
        
        for i in range(self.num_layers_decoder):
          query = self.time_series_model_query[i](query)  
          if self.roi_feature_encoder:
            query = self.feature_intra[i](query, blink_features)
         
        #score = self.score_head(query).squeeze(-1)  # (T,)
        
        for layer in self.score_head[:-1]:
          query = layer(query)
        
        score = self.score_head[-1](query)
        
        if return_query:
          return score, query
        else:
          return score


# Positional Encoding 实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)
