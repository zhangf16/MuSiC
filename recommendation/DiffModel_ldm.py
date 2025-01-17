import torch
import torch.nn as nn
import math

def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    时间步嵌入。
    """
    timesteps = timesteps.to(dtype=torch.float32)
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

def nonlinearity(x):
    # Swish激活函数
    return x * torch.sigmoid(x)

def NormalizeVector(in_features):
    return nn.LayerNorm(in_features)

class AttentionBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.norm = NormalizeVector(in_features)
        self.qkv = nn.Linear(in_features, in_features * 3)
        self.proj_out = nn.Linear(in_features, in_features)
    
    def forward(self, x):
        """
        x: (batch_size, in_features)
        """
        h = self.norm(x)
        # 计算 q, k, v
        qkv = self.qkv(h)  # (batch_size, in_features * 3)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)  # 每个都是 (batch_size, in_features)
        
        # 调整形状以在特征维度上计算注意力
        q = q.unsqueeze(1)  # (batch_size, 1, in_features)
        k = k.unsqueeze(1)  # (batch_size, 1, in_features)
        v = v.unsqueeze(1)  # (batch_size, 1, in_features)
        
        # 计算注意力权重
        attn_weights = torch.bmm(q, k.transpose(1, 2))  # (batch_size, 1, 1)
        attn_weights = attn_weights * (self.in_features ** -0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # 应用注意力到值
        attn_output = torch.bmm(attn_weights, v)  # (batch_size, 1, in_features)
        attn_output = attn_output.squeeze(1)  # (batch_size, in_features)
        
        h = self.proj_out(attn_output)
        return x + h

class ResnetBlock(nn.Module):
    def __init__(self, *, in_features, out_features=None, dropout, temb_features):
        super().__init__()
        self.in_features = in_features
        out_features = in_features if out_features is None else out_features
        self.out_features = out_features

        self.norm1 = NormalizeVector(in_features)
        self.fc1 = nn.Linear(in_features, out_features)
        self.temb_proj = nn.Linear(temb_features, out_features)
        self.norm2 = NormalizeVector(out_features)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_features, out_features)

        # 如果输入和输出维度不同，需要额外的调整层
        if self.in_features != self.out_features:
            self.adjust_fc = nn.Linear(in_features, out_features)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.fc1(h)
        h = h + self.temb_proj(nonlinearity(temb))  # (batch_size, out_features)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.fc2(h)

        # 如果输入和输出维度不同，使用调整层
        if self.in_features != self.out_features:
            x = self.adjust_fc(x)

        return x + h

class DiffCDR(nn.Module):
    def __init__(self, num_steps, t, in_features, hidden_dims, dropout=0.0, attn_resolutions=[]):
        super().__init__()

        self.num_steps = num_steps
        self.t = t
        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_prod_p = torch.cat([torch.tensor([1.0]), self.alphas_prod[:-1]], 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        self.in_features = in_features
        self.temb_features = in_features * 4  # 调整时间嵌入维度为 in_features 的 4 倍

        # 时间嵌入
        self.temb = nn.Sequential(
            nn.Linear(self.temb_features, self.temb_features),
            nn.SiLU(),
            nn.Linear(self.temb_features, self.temb_features),
        )

        self.fc_down = nn.ModuleList()
        self.down_res_blocks = nn.ModuleList()
        self.down_attn_blocks = nn.ModuleList()
        self.hs_dims = []
        prev_features = in_features

        # 下采样
        for i, dim in enumerate(hidden_dims):
            self.fc_down.append(nn.Linear(prev_features, dim))
            self.down_res_blocks.append(ResnetBlock(
                in_features=dim, out_features=dim, dropout=dropout, temb_features=self.temb_features
            ))
            # if i in attn_resolutions:
            #     self.down_attn_blocks.append(AttentionBlock(dim))
            # else:
            #     self.down_attn_blocks.append(nn.Identity())
            self.down_attn_blocks.append(AttentionBlock(dim))
            self.hs_dims.append(dim)
            prev_features = dim

        # 反转 hs_dims
        self.hs_dims = self.hs_dims[::-1]

        # 中间层
        self.mid_block1 = ResnetBlock(
            in_features=prev_features, out_features=prev_features, dropout=dropout, temb_features=self.temb_features
        )
        self.mid_attn = AttentionBlock(prev_features)
        self.mid_block2 = ResnetBlock(
            in_features=prev_features, out_features=prev_features, dropout=dropout, temb_features=self.temb_features
        )

        # 上采样
        self.fc_up = nn.ModuleList()
        self.up_res_blocks = nn.ModuleList()
        self.up_attn_blocks = nn.ModuleList()
        hidden_dims_reverse = list(reversed(hidden_dims))

        for i, dim in enumerate(hidden_dims_reverse):
            h_dim = self.hs_dims[i]

            self.fc_up.append(nn.Linear(prev_features, dim))

            in_features_res = dim + h_dim

            self.up_res_blocks.append(ResnetBlock(
                in_features=in_features_res, out_features=dim, dropout=dropout, temb_features=self.temb_features
            ))

            # if i in attn_resolutions:
            #     self.up_attn_blocks.append(AttentionBlock(dim))
            # else:
            #     self.up_attn_blocks.append(nn.Identity())
            self.up_attn_blocks.append(AttentionBlock(dim))
            prev_features = dim

        # 输出层
        self.fc_out = nn.Linear(prev_features, self.in_features)

    def forward(self, x, t, context=None):
        if context is not None:
            x = x + context

        # 获取时间嵌入
        temb = get_timestep_embedding(t, self.temb_features)
        temb = self.temb(temb)

        # 保存每层的特征以用于跳跃连接
        hs = []
        # 下采样
        for fc, res_block, attn_block in zip(self.fc_down, self.down_res_blocks, self.down_attn_blocks):
            x = fc(x)
            x = res_block(x, temb)
            x = attn_block(x)
            hs.append(x)

        # 中间层
        x = self.mid_block1(x, temb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, temb)

        # 上采样
        for i, (fc, res_block, attn_block) in enumerate(zip(self.fc_up, self.up_res_blocks, self.up_attn_blocks)):
            x = fc(x)
            h = hs.pop()
            x = torch.cat([x, h], dim=-1)
            x = res_block(x, temb)
            x = attn_block(x)

        # 输出层
        x = self.fc_out(x)
        return x


# 其他函数保持不变


import torch.nn.functional as F

def q_x_fn(model,x_0,t,device):
    #eq(4)
    noise = torch.normal(0,1,size = x_0.size() ,device=device)

    alphas_t = model.alphas_bar_sqrt.to(device)[t]
    alphas_1_m_t = model.one_minus_alphas_bar_sqrt.to(device)[t]

    return (alphas_t * x_0 + alphas_1_m_t * noise),noise

def diffusion_loss(model, x_0, device,cond_emb=None):

    num_steps = model.num_steps

    batch_size = x_0.shape[0]
    #sample t
    # t = torch.randint(0,num_steps,size=(batch_size//2,),device=device)
    # if batch_size%2 ==0:
    #     t = torch.cat([t,num_steps-1-t],dim=0)
    # else:
    #     extra_t = torch.randint(0,num_steps,size=(1,),device=device)
    #     t = torch.cat([t,num_steps-1-t,extra_t],dim=0)
    
    t = torch.randint(0, num_steps, size=(batch_size,), device=device)
    t = t.unsqueeze(-1)

    x,e = q_x_fn(model,x_0,t,device)
    output = model(x, t.squeeze(-1), cond_emb)

    # first_noising_step = (t == 0)
    # x_minus_1 = torch.where(first_noising_step, x_0, q_x_fn(model, x_0, t - 1, device)[0])

    return F.smooth_l1_loss(e, output, reduction='none')
    # return F.smooth_l1_loss(x_minus_1, output)

def p_sample(model, x0, device, cond_emb=None):
    steps = model.num_steps
    sampling_steps = model.t
    if sampling_steps == 0:
        x = x0
    else:
        x,e = q_x_fn(model, x0, sampling_steps-1, device)
    # x = torch.normal(0,1,size = x0.size() ,device=device)
    for step in range(steps):
        # Compute the corresponding timestep
        t = torch.full((x.shape[0],), steps - step - 1, dtype=torch.long, device=device)
        predicted_noise = model(x, t, cond_emb)

        # 从预测的噪声中恢复 X_{t-1}
        alpha_t = model.alphas_bar_sqrt.to(device)[t].unsqueeze(-1)
        alpha_1_m_t = model.one_minus_alphas_bar_sqrt.to(device)[t].unsqueeze(-1)
        beta_t = model.betas.to(device)[t].unsqueeze(-1)
        x = (x - predicted_noise * beta_t / alpha_1_m_t) / alpha_t

        # z = torch.rand_like(x)
        # x = beta_t.sqrt() * z + x
    return x

def p_sample_random(model, x0, device, cond_emb=None):
    steps = model.num_steps
    x = torch.normal(0,1,size = x0.size() ,device=device)
    for step in range(steps):
        # Compute the corresponding timestep
        t = torch.full((x.shape[0],), steps - step - 1, dtype=torch.long, device=device)
        predicted_noise = model(x, t, cond_emb)

        # 从预测的噪声中恢复 X_{t-1}
        alpha_t = model.alphas_bar_sqrt.to(device)[t].unsqueeze(-1)
        alpha_1_m_t = model.one_minus_alphas_bar_sqrt.to(device)[t].unsqueeze(-1)
        beta_t = model.betas.to(device)[t].unsqueeze(-1)
        x = (x - predicted_noise * beta_t / alpha_1_m_t) / alpha_t

        # z = torch.rand_like(x)
        # x = beta_t.sqrt() * z + x
    return x
