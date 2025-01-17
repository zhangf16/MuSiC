import torch
import torch.nn as nn

import math

from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

noise_schedule = NoiseScheduleVP(schedule='linear')


#---------------------------------------------------------
def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    timesteps = timesteps.to(dtype=torch.float32)

    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    assert embedding_dim % 2 == 0
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def NormalizeVector(in_features):
    return torch.nn.LayerNorm(in_features)

class Upsample(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return nonlinearity(self.fc(x))

class Downsample(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return nonlinearity(self.fc(x))
    
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
        # print(h.shape)

        # h = h + nonlinearity(temb)
        h = h + self.temb_proj(nonlinearity(temb))

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.fc2(h)

        # 如果输入和输出维度不同，使用调整层
        if self.in_features != self.out_features:
            x = self.adjust_fc(x)

        return x + h
    
class DiffCDR(nn.Module):
    def __init__(self, num_steps, t, in_features, hidden_dims, dropout=0.0):
        super().__init__()

        #define params
        self.num_steps = num_steps
        self.t = t
        self.betas = torch.linspace(1e-4,0.02 ,num_steps)
        self.alphas = 1-self.betas
        self.alphas_prod = torch.cumprod(self.alphas,0)
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(),self.alphas_prod[:-1]],0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        assert self.alphas.shape==self.alphas_prod.shape==self.alphas_prod_p.shape==\
        self.alphas_bar_sqrt.shape==self.one_minus_alphas_bar_log.shape\
        ==self.one_minus_alphas_bar_sqrt.shape

        self.in_features = in_features
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(in_features, in_features),
            nn.Linear(in_features, in_features),
        ])

        self.fc_down = nn.ModuleList()
        self.down_res_blocks = nn.ModuleList()  # 新增残差块列表
        prev_features = in_features
        for dim in hidden_dims:
            self.fc_down.append(nn.Linear(prev_features, dim))
            # 每个下采样步骤后添加残差块
            self.down_res_blocks.append(ResnetBlock(in_features=dim, out_features=dim, 
                                                    dropout=dropout, temb_features = in_features))
            prev_features = dim


        # 中间残差块
        self.mid_block = ResnetBlock(
            in_features=prev_features, out_features=prev_features, 
            dropout=dropout, temb_features = in_features)
        
        self.fc_up = nn.ModuleList()
        self.up_res_blocks = nn.ModuleList()  # 新增残差块列表
        for dim in reversed(hidden_dims):
            self.fc_up.append(nn.Linear(prev_features, dim))
            # 每个下采样步骤后添加残差块
            self.up_res_blocks.append(ResnetBlock(in_features=dim, out_features=dim, 
                                                  dropout=dropout, temb_features = in_features))
            prev_features = dim


        # 输出层
        self.fc_out = nn.Linear(prev_features, in_features)


        self.layer1 = nn.Linear(in_features, in_features*2)
        self.tanh = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)  # 添加Dropout层
        self.output_layer = nn.Linear(in_features*2, in_features)

    def forward(self, x, t, context=None):
        if context is not None:
            x = x + context
        temb = get_timestep_embedding(t, self.in_features)


        # x = x + temb
        # x = self.tanh(self.layer1(x))
        # x = self.output_layer(x)
        # return x
    
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # 下采样
        for fc, res_block in zip(self.fc_down, self.down_res_blocks):
            x = fc(x)
            x = res_block(x, temb)

        # 中间层
        x = self.mid_block(x, temb)

        # 上采样
        for fc, res_block in zip(self.fc_up, self.up_res_blocks):
            x = fc(x)
            x = res_block(x, temb)

        # 输出层
        x = self.fc_out(x)
        return x



#---------------------------------------------------------
#loss 
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

#generation fun
# def p_sample(model, x0, device, cond_emb=None):
#     x = torch.normal(0,1,size = cond_emb.size() ,device=device)
#     #wrap for dpm_solver
#     classifier_scale_para = 1.0
#     dmp_sample_steps = 30
#     num_steps = model.num_steps

#     model_kwargs ={'cond_emb':cond_emb,
#                 'cond_mask':torch.zeros( cond_emb.size()[0] ,device=device),
#                 }


#     model_fn = model_wrapper(
#         model,
#         noise_schedule,
#         is_cond_classifier=True,
#         classifier_scale = classifier_scale_para, 
#         time_input_type="1",
#         total_N=num_steps,
#         model_kwargs=model_kwargs
#     )

#     dpm_solver = DPM_Solver(model_fn, noise_schedule)

#     sample = dpm_solver.sample(
#                     x,
#                     steps=dmp_sample_steps,
#                     eps=1e-4,
#                     adaptive_step_size=False,
#                     fast_version=True,
#                 )
    
#     return sample.to(device)