import torch
import math

def convert_psk(tensor, categories):
    device = tensor.device
    tensor = tensor.int()
    cat_spacings = 2 * math.pi / torch.tensor(categories).to(device)
    
    phases = tensor * cat_spacings
    psk_enc = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)
    return psk_enc

def psk2cat(psk_tensor, categories):
    device = psk_tensor.device
    cat_len = categories.shape[0]
    cat_spacings = 2 * math.pi / torch.tensor(categories).to(device)
    
    sin_phases = psk_tensor[:, :cat_len]
    cos_phases = psk_tensor[:, cat_len:2*cat_len]
    
    phases = torch.atan2(sin_phases, cos_phases)
    phases = (phases + 2 * math.pi) % (2 * math.pi)
    
    feature_cat = torch.round(phases / cat_spacings).long()

    total = feature_cat.numel()
    corrected = 0

    for i, cat in enumerate(categories):
        corrected += (feature_cat[:, i] >= cat).sum().item()
        feature_cat[:, i] = torch.where(feature_cat[:, i] >= cat, 0, feature_cat[:, i])

    if total > 0:
        casting_rate = corrected / total
        print(f"Casting rate in psk2cat: {round(casting_rate, 3)}")
    
    return feature_cat