# Modeling the Lighting as Style Factor via Neural Networks for White Balance Correction

## Overview
This repository contains the implementation of multiple white balance correction approaches, developed during my PhD research. Our work spans several papers, with following focuses:

- Treating white balance as a style factor (Style WB)
- Deterministic Neural Illumination Mapping (DeNIM)
- Transformer-based architecture with feature distribution matching to represent the style factor (FDM WB)
- Feature distribution matching as objective function for white balance correction (FDM Loss)

## Research Idea
`Any disruptive factor in an image can be modeled as style factor.`

`Lighting and illumination are disruptive factors in an image.`

## Approaches
This repository includes three main implementations for white balance correction:

1. **Style-Based White Balance (style_wb.py)**
   - Treats lighting as a style factor
   - Uses IFRNet variant with VGG feature extraction
   - Single style removal option: AdaIN
   - Produces weights for blending multiple white balance settings

2. **Deterministic Neural Illumination Mapping (denim.py)**
   - More efficient modeling of lighting as a style factor
   - Uses Style WB as backbone
   - Color-mapping based approach with resolution-agnostic modeling

3. **Feature Distribution Matching (fdm_wb.py)**
   - More advanced modeling of lighting as a style factor
   - Window-based self-attention mechanisms
   - Multiple style removal options including EFDM and AdaIN
   - Produces weights for blending multiple white balance settings

4. **FDM Loss Functions (fdm_loss.py)**
   - Specialized loss functions for training white balance correction networks
   - Uses Vision Transformer (ViT) to extract features
   - Combines self-similarity, identity, and CLS token losses
   - Focuses on making the network robust to multi-illuminant scenarios


## Usage

### Style-Based White Balance Correction Network (Style WB)
```python
from style_wb import IFRNetv2
import torch
from torchvision.models import vgg16

# Load VGG features
vgg_feats = vgg16(pretrained=True).features[:35].eval()

# Create model (5 WB settings with 3 channels each)
model = IFRNetv2(
    vgg_feats=vgg_feats, 
    in_channels=15,  # 5 settings x 3 channels
    base_n_channels=32, 
    input_size=64, 
    out_channels=5   # Output weights for 5 settings
)

# Process images and apply weighted blending
weights = model(input_images)
weights = torch.clamp(weights, -1000, 1000)
weights = nn.Softmax(dim=1)(weights)

# Blend the weights with their corresponding WB settings
corrected_image = torch.unsqueeze(weights[:, 0, :, :], dim=1) * input_images[:, :3, :, :]
for i in range(1, input_images.shape[1] // 3):
    corrected_image += torch.unsqueeze(weights[:, i, :, :], dim=1) * input_images[:, (i * 3):3 + (i * 3), :, :]
```

### Deterministic Neural Illumination Mapping (DeNIM)
```python
from denim import DeNIM
import torch
k, ch, sz, ps = 32, 9, 256, 64
resolution = 2048  # arbitrary resolution
dncm_to_canon = DeNIM_to_Canon(k, ch).cuda()
dncm_wo_fusion = DeNIM_wo_Fusion(k).cuda()
awb_encoder = AWBEncoder(sz, k, ch, ps, None).cuda()

I = torch.randn(1, ch, resolution, resolution).cuda()
with torch.no_grad():
    d = awb_encoder(I)
    canonical_image = dncm_to_canon(I, d)
    corrected_image = dncm_wo_fusion(canonical_image)
```


### Feature Distribution Matching for White Balance Correction (FDM WB)
```python
from fdm_wb import StyleUformer
import torch
from torchvision.models import vgg16

# Load VGG features for style extraction
vgg_feats = vgg16(pretrained=True).features[:35].eval()

# Create model
model = StyleUformer(
    vgg_feats=vgg_feats,
    img_size=256,
    in_chans=9,  # 3 RGB images with different white balance
    embed_dim=32,
    win_size=8,
    style_remover="efdm"  # Options: "efdm", "adain"
)

# Process images and apply weighted blending
weights = model(input_images)
weights = torch.clamp(weights, -1000, 1000)
weights = nn.Softmax(dim=1)(weights)

# Blend the weights with their corresponding WB settings
corrected_image = torch.unsqueeze(weights[:, 0, :, :], dim=1) * input_images[:, :3, :, :]
for i in range(1, input_images.shape[1] // 3):
    corrected_image += torch.unsqueeze(weights[:, i, :, :], dim=1) * input_images[:, (i * 3):3 + (i * 3), :, :]
```

### Feature Distribution Matching as Objective Function (FDM Loss)
```python
from fdm_loss import FDMLoss

# Initialize loss function
loss_fn = FDMLoss(white_level=1.0)  # white_level is the maximum pixel value

# During training
predicted = model(input_images)
loss_dict = loss_fn(predicted, {
    'input_rgb': input_images, 
    'gt_uv': ground_truth_uv,
    'step': current_step
})

# Total loss
total_loss = loss_dict['loss']
```

## Citations
If you find this work useful for your research, please cite our papers:

**Style WB:**
```
@InProceedings{Kinli2023stylewb,
    author    = {Kınlı, Furkan and Yılmaz, Doğa and Özcan, Barış and Kıraç, Furkan},
    title     = {Modeling the Lighting in Scenes As Style for Auto White-Balance Correction},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {4903-4913}
}
```

**DeNIM:**
```
@InProceedings{Kinli2023denim,
    author    = {K{\i}nl{\i}, Furkan and Y{\i}lmaz, Do\u{g}a and \"Ozcan, Bar{\i}\c{s} and K{\i}ra\c{c}, Furkan},
    title     = {Deterministic Neural Illumination Mapping for Efficient Auto-White Balance Correction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2023},
    pages     = {1139-1147}
}
```

**FDM WB:**
```
@article{Kinli2025fdmwb,
    title   = {Advancing white balance correction through deep feature statistics and feature distribution matching},
    journal = {Journal of Visual Communication and Image Representation},
    volume  = {108},
    pages   = {104412},
    year    = {2025},
    issn    = {1047-3203},
    doi     = {https://doi.org/10.1016/j.jvcir.2025.104412},
    url     = {https://www.sciencedirect.com/science/article/pii/S1047320325000264},
    author  = {Furkan Kınlı and Barış Özcan and Furkan Kıraç},
}
```

**FDM Loss:**
```
@article{Kinli2025fdmloss,
  author    = {Kınlı, Furkan and Kıraç, Furkan},
  title     = {Feature distribution statistics as a loss objective for robust white balance correction},
  journal   = {Machine Vision and Applications},
  year      = {2025},
  volume    = {36},
  number    = {3},
  pages     = {58},
  doi       = {10.1007/s00138-025-01680-1},
  url       = {https://doi.org/10.1007/s00138-025-01680-1},
  issn      = {1432-1769},
}
```


