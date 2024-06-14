import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.0.0'):
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()

from vit_pytorch.simple_vit_flexatt import SimpleViTflexAtt


def check_vit_names(name):
    for n in ['SimpleViTS']:
        if name.startswith(n):
            return True
    return False

def get_vit_model(model, num_classes=1000, **kwargs):
    if model.startswith('SimpleViTS'):
        return SimpleViTflexAtt(
                    image_size = 224,
                    patch_size = 16,
                    num_classes = num_classes,
                    dim = 384,
                    depth = 12,
                    heads = 6,
                    mlp_dim = 1536,
                    dim_head=64,
                    att=model.split('_')[1] #e.g., SimpleViTS_msf-g2
                )