
import torch

def calc_dummy_node_loss_loop(input_nodes, attn_weight, num_original_nodes):
    batch_size, num_nodes, feature_dim = input_nodes.size()
    dummy_node_indices = list(range(num_original_nodes+1, num_nodes))
    dummy_node_loss = 0.0
    for i in dummy_node_indices:
        dummy_node_loss += (attn_weight[:, :, i, :].abs().sum(dim=-1)).mean()
        dummy_node_loss += (attn_weight[:, :, :, i].abs().sum(dim=-1)).mean()
    dummy_node_loss /= len(dummy_node_indices) * 2
    dummy_node_loss /= num_nodes
    return dummy_node_loss

def calc_dummy_node_loss_vectorized(input_nodes, attn_weight, num_original_nodes):
    batch_size, num_nodes, feature_dim = input_nodes.size()
    start_idx = num_original_nodes + 1
    
    # attn_weight: [batch, heads, nodes, nodes]
    
    # Rows: [batch, heads, dummy_nodes, nodes]
    dummy_rows = attn_weight[:, :, start_idx:, :]
    
    # Cols: [batch, heads, nodes, dummy_nodes]
    dummy_cols = attn_weight[:, :, :, start_idx:]
    
    # Sum of absolute values
    row_sum = dummy_rows.abs().sum()
    col_sum = dummy_cols.abs().sum()
    
    # The loop implementation sums row_sum and col_sum.
    # It counts the intersection (dummy-dummy) twice.
    # The user's proposal "total = dummy_node_attention + dummy_node_attention_col - dummy_node_attention_dup"
    # suggests they want to count intersection once.
    
    # Let's replicate the LOOP behavior first to see if we can vectorize it exactly.
    # The loop sums (row_i_sum + col_i_sum) for each i.
    # This is exactly row_sum + col_sum.
    
    total_sum = row_sum + col_sum
    
    # Normalization in loop:
    # mean() is taken over (batch * heads).
    # dummy_node_loss += (attn_weight[:, :, i, :].abs().sum(dim=-1)).mean()
    # .mean() here divides by (batch * heads).
    
    # So loop computes:
    # Sum_i [ (Sum_row_i / (B*H)) + (Sum_col_i / (B*H)) ]
    # = (Sum_all_rows + Sum_all_cols) / (B*H)
    
    num_dummy = num_nodes - start_idx
    if num_dummy == 0:
        return 0.0
        
    loss = total_sum / (batch_size * attn_weight.size(1))
    loss /= (num_dummy * 2)
    loss /= num_nodes
    
    return loss

# Setup
B, H, N, D = 2, 4, 10, 16
num_original = 6
input_nodes = torch.randn(B, N, D)
attn_weight = torch.randn(B, H, N, N)

loss_loop = calc_dummy_node_loss_loop(input_nodes, attn_weight, num_original)
loss_vec = calc_dummy_node_loss_vectorized(input_nodes, attn_weight, num_original)

print(f"Loop loss: {loss_loop}")
print(f"Vec loss: {loss_vec}")
print(f"Match: {torch.allclose(torch.tensor(loss_loop), torch.tensor(loss_vec))}")
