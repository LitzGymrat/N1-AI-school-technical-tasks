import torch
import torch.nn as nn
from transformer.RMSNorm_module import RMSNorm
from transformer.Multihead_self_attention_module import Multihead_self_attention
from moe.MoE_module import MoE

class Transformer_blk(nn.Module):
    # d_ff 是配置指定的单个专家宽度。
    def __init__(self, d_model:int, num_heads:int,d_ff:int,max_seq_len:int,theta:float,
                 num_experts:int,top_k:int,device:torch.device|None = None, dtype:torch.dtype | None =None,
                 norm_position:str = "pre", use_rope:bool = True):

        super().__init__()
        self.d_in = d_model
        self.device = device
        self.dtype = dtype
        self.d_ff = d_ff
        self.num_heads = num_heads
        if norm_position not in ("pre", "post", "none"):
            raise ValueError(f"Unknown norm_position: {norm_position}")
        self.norm_position = norm_position
        #两个子层的rmsnorm各自参数独立，需要有两个
        self.rmsnorm_1 = nn.Identity() if norm_position == "none" else RMSNorm(d_model,device = device,dtype = dtype)
        self.rmsnorm_2 = nn.Identity() if norm_position == "none" else RMSNorm(d_model,device = device,dtype = dtype)
        # 保留原 block 的注意力与残差结构，仅把 FFN 换成 MoE。
        self.moe = MoE(d_model,d_ff,num_experts,top_k,device=device,dtype=dtype)
        self.msa = Multihead_self_attention(d_model,num_heads,theta,max_seq_len,device=device,dtype=dtype,use_rope=use_rope)


    def forward(self,x:torch.Tensor,token_positions:torch.Tensor|None = None) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        if self.norm_position == "post":
            y_1 = self.rmsnorm_1(x + self.msa(x,token_positions))
            ffn_output,balance_loss,expert_load = self.moe(y_1)
            return self.rmsnorm_2(y_1 + ffn_output),balance_loss,expert_load
        y_1 = x + self.msa(self.rmsnorm_1(x),token_positions)
        ffn_output,balance_loss,expert_load = self.moe(self.rmsnorm_2(y_1))
        y_2 = y_1 + ffn_output
        # 辅助损失直接传给 LM；不 detach，否则 router 收不到这部分梯度。
        return y_2,balance_loss,expert_load
