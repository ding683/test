from segmentation import segmentation
from skimage.measure import regionprops, label
from tifffile import imread, imwrite
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from resnet_custom import model_baseline
import pandas as pd
def get_nucleus(mask,image):
# 识别单独的细胞核区域
 nuclei_mask =( mask != 0)
 nucleus_image = image * nuclei_mask[:, :, np.newaxis]
# 将掩码中的每个细胞标记为独立区域
 regions = regionprops(mask)
# 保存每个细胞核的ROI
 rois = []
 for region in regions:
    min_row, min_col, max_row, max_col = region.bbox  # 获取细胞核的边界框
    roi = nucleus_image[min_row:max_row, min_col:max_col]# 裁剪出单个细胞核
    rois.append(roi)
 return rois ,regions

def features_extract(mode,rois,size,means,stds): 
 model = model_baseline(model_name=mode, pretrained=True)
# gpu加速
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
#  model1 = model1.to(device)
#  model1.eval()  
#  transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#  ])
#  features = []
#  for roi in rois:
#     input_tensor = transform(roi).unsqueeze(0).to(device)  # 转换为4维张量
#     with torch.no_grad():
#         feature1 = model1(input_tensor)  # 提取特征
#         feature=feature1.cpu().numpy()
#         features.append(feature)
# 加载 ResNet 模型
 model.eval()  # 设为评估模式
# 定义图像预处理
 transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=means, std=stds)
  ])

# 提取深度学习特征
 features = []
 for roi in rois:
    input_tensor = transform(roi).unsqueeze(0)  # 转换为4维张量
    with torch.no_grad():
        feature = model(input_tensor).cpu().numpy()  # 提取特征
        features.append(feature)
 return features

#将获得的特征存储
def save_features(features,regions):
# 整合特征
 data = []
 for i in range(len(features)):
    feature_vector = features[i].flatten()  # 展平深度学习特征向量
    data.append({**{'id':f'ID_{regions[i].label}'},**{f'Feature_{j}': feature_vector[j] for j in range(len(feature_vector))}})
# 保存到 CSV 文件
 df = pd.DataFrame(data)
 df.to_csv('cell_features.csv', index=False)
