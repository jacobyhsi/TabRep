import torch

def feature_to_bits(feature_values, num_bits):
    """Converts a tensor of feature values to a tensor of bit representations."""
    device = feature_values.device
    mask = 2**torch.arange(num_bits - 1, -1, -1, device=device, dtype=torch.long)
    bits = ((feature_values.unsqueeze(-1) & mask) != 0).float()
    bits = bits * 2 - 1  # Normalize to -1 to 1
    return bits

def convert_categorical_to_bits(tensor, num_bits_per_feature):
    """Converts a tensor of categorical features into a tensor of bit representations,
       including calculating the maximum bit length needed for each feature."""
    tensor = tensor.int()

    bit_tensors = []
    for i, num_bits in enumerate(num_bits_per_feature):
        feature_bits = feature_to_bits(tensor[:, i], num_bits)
        bit_tensors.append(feature_bits)

    # Concatenate all bit representations along the last dimension
    final_bit_tensor = torch.cat(bit_tensors, dim=-1)
    return final_bit_tensor

def bits_to_categorical(bits_tensor, num_bits_per_feature):
    """Converts a tensor of bit representations back to categorical features.
    
    Args:
        bits_tensor: A tensor containing bit representations.
        num_bits_per_feature: A list indicating the number of bits for each feature.
    
    Returns:
        A tensor of categorical features.
    """
    device = bits_tensor.device
    start_index = 0
    categorical_features = []
    
    # Convert bits from -1 to 1 range back to 0 and 1
    bits_tensor = ((bits_tensor + 1) / 2).int()
    print(num_bits_per_feature)
    
    for num_bits in num_bits_per_feature:
        # Extract bits for the current feature
        end_index = start_index + num_bits
        feature_bits = bits_tensor[:, start_index:end_index]
        
        # Convert bits to decimal
        mask = 2 ** torch.arange(num_bits - 1, -1, -1, device=device, dtype=torch.long)
        categorical_feature = torch.sum(feature_bits * mask, dim=1)
        
        categorical_features.append(categorical_feature.unsqueeze(-1))
        start_index = end_index
    
    # Concatenate all categorical features along the last dimension
    final_categorical_tensor = torch.cat(categorical_features, dim=-1)
    return final_categorical_tensor
