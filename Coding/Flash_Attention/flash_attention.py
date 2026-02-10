# softmax
def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=-1, keepdim=True)

# safe softmax
def safe_softmax(x):
    max_x = torch.max(x, dim=-1, keepdim=True)
    return torch.exp(x - max_x) / torch.sum(torch.exp(x - max_x), dim=-1, keepdim=True)

# online softmax
def online_softmax(x):
    pre_max = x[:-1].max()
    pre_sum = torch.sum(torch.exp(x[:-1] - pre_max))
    cur_max = x.max()
    cur_sum = pre_sum * torch.exp(pre_max - cur_max) + torch.exp(x[-1] - cur_max)
    return torch.exp(x - cur_max) / cur_sum

# multi-block online softmax
def block_online_softmax(x, block_size = 3):
    # 区分三个block
    blocks = torch.split(x, block_size)
    x_blocks_0_max = blocks[0].max()
    x_blocks_0_sum = torch.sum(torch.exp(blocks[0] - x_blocks_0_max))
    x_blocks_1_max = blocks[1].max()
    x_blocks_1_sum = torch.sum(torch.exp(blocks[1] - x_blocks_1_max))
    x_blocks_2_max = blocks[2].max()
    x_blocks_2_sum = torch.sum(torch.exp(blocks[2] - x_blocks_2_max))
    
    M = [x_blocks_0_max, x_blocks_1_max, x_blocks_2_max]
    S = [x_blocks_0_sum, x_blocks_1_sum, x_blocks_2_sum]
    
    M_old = torch.tensor([0.0])
    S_old = torch.tensor([0.0])
    for i in range(block_size):
        M_new = torch.max(M[i], M_old)
        S_new = S[i] * torch.exp(M_old - M_new) + torch.exp(blocks[i] - M_new).sum()
        M_old = M_new
        S_old = S_new

    return torch.exp(x - M_new) / S_new

# 完全参考 https://zhuanlan.zhihu.com/p/663932651
def flash_attention(Q, K, V, block_size = 3):
    B, H, N, d = Q.shape
    O = torch.zeros_like(Q)

    m = torch.full((B, H, N, 1), -1e10, device=Q.device)
    l = torch.zeros((B, H, N, 1), device=Q.device)

    Q_blocks = torch.split(Q, block_size, dim=2)
    K_blocks = torch.split(K, block_size, dim=2)
    V_blocks = torch.split(V, block_size, dim=2)

    m_blocks = list(torch.split(m, block_size, dim=2))
    l_blocks = list(torch.split(l, block_size, dim=2))
    O_blocks = list(torch.split(O, block_size, dim=2))

    for j in range(len(K_blocks)):
        Kj = K_blocks[j]
        Vj = V_blocks[j]
        for i in range(len(Q_blocks)):
            Qi = Q_blocks[i]
            mi = m_blocks[i]
            li = l_blocks[i]
            Oi = O_blocks[i]
            S = Qi @ Kj.transpose(-2, -1)      # [B,H,bi,bj]
            m_ij, _  = torch.max(S, dim=-1, keepdim=True)
            P = torch.exp(S - m_ij)
            l_ij = torch.sum(P, dim=-1, keepdim=True)

            m_new = torch.maximum(mi, m_ij)
            li_new = torch.exp(mi - m_new) * li + torch.exp(m_ij - m_new) * l_ij

            Oi = torch.exp(mi - m_new) * (li / li_new) * Oi + torch.exp(m_ij - m_new) / li_new * (P @ Vj)


            O_blocks[i] = Oi
            m_blocks[i] = m_new
            l_blocks[i] = li_new

    return torch.cat(O_blocks, dim=2)


def self_attention(Q, K, V, block_size = 3):
    return torch.softmax(Q @ K.transpose(2,3), dim = -1) @ V

def flash_attention2(Q, K, V, block_size = 3):
    B, H, N, d = Q.shape
    O = torch.zeros_like(Q)

    m = torch.full((B, H, N, 1), -1e10, device=Q.device)
    l = torch.zeros((B, H, N, 1), device=Q.device)

    Q_blocks = torch.split(Q, block_size, dim=2)
    K_blocks = torch.split(K, block_size, dim=2)
    V_blocks = torch.split(V, block_size, dim=2)

    m_blocks = list(torch.split(m, block_size, dim=2))
    l_blocks = list(torch.split(l, block_size, dim=2))
    O_blocks = list(torch.split(O, block_size, dim=2))

    for i in range(len(Q_blocks)):
        Qi = Q_blocks[i]
        mi = m_blocks[i]
        li = l_blocks[i]
        Oi = O_blocks[i]

        for j in range(len(K_blocks)):
            Kj = K_blocks[j]
            Vj = V_blocks[j]

            S = Qi @ Kj.transpose(-2, -1)      # [B,H,bi,bj]
            m_ij, _ = torch.max(S, dim=-1, keepdim=True)
            P = torch.exp(S - m_ij)
            l_ij = torch.sum(P, dim=-1, keepdim=True)

            m_new = torch.maximum(mi, m_ij)
            li = torch.exp(mi - m_new) * li + torch.exp(m_ij - m_new) * l_ij
            Oi = torch.exp(mi - m_new) * Oi + torch.exp(m_ij - m_new) * (P @ Vj)

            mi = m_new

        O_blocks[i] = Oi / li
        m_blocks[i] = mi
        l_blocks[i] = li

    return torch.cat(O_blocks, dim=2)

import torch

if __name__ == "__main__":
    Q = torch.randn(1, 1, 9, 10)
    K = torch.randn(1, 1, 9, 10)
    V = torch.randn(1, 1, 9, 10)
    # print(flash_attention(Q, K, V))
    # print(flash_attention2(Q, K, V))
    # print(self_attention(Q, K, V))
    standard_attention = self_attention(Q, K, V)
    attention = flash_attention(Q, K, V)
    attention2 = flash_attention2(Q, K, V)
    print(f"diff {standard_attention - attention}")
    print(f"diff {standard_attention - attention2}")