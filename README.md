# TGCMFNet: Text-Guided Cross-Modal Fusion Network

**TGCMFNet** (Text-Guided Cross-Modal Fusion Network) 是一个用于高光谱图像（HSI）和LiDAR数据融合分类的深度学习框架。该网络通过引入文本引导的跨模态对齐机制，实现了多模态数据的高效融合和精确分类。


## 🎯 项目简介

TGCMFNet 提出了一种新颖的文本引导跨模态融合方法，用于HSI和LiDAR数据的联合分类。该方法的核心创新包括：

- **跨模态交互模块（CMIM）**：实现HSI和LiDAR之间的深度交互和信息融合
- **多模态桥接对齐模块（MBAM）**：通过对比学习实现不同模态特征的对齐
- **视觉-文本对齐模块（VTAM）**：结合CLIP实现视觉特征与文本特征的语义对齐



### 使用示例

```bash
# 使用Houston 2013数据集，每类20个样本
python TGCMFNet/demo.py --dataset houston2013 --num_labelled 20 --epoches 300
```

### 结果保存

- **模型权重**：保存在 `TGCMFNet/log/{dataset_name}.pkl`
- **训练日志**：控制台输出详细的训练过程

## 📚 项目结构

```
TGCMFNet/
├── demo.py                 # 主训练脚本
├── TGCMFNet.py            # 网络模型定义（主模型）
├── CMIM_MBAM.py           # 跨模态交互模块
├── utils.py               # 工具函数（数据加载、评估等）
├── clip.py                # CLIP模型实现
├── simple_tokenizer.py    # 文本分词器
├── requirements.txt       # 依赖包列表
├── ViT-B-32.pt           # CLIP预训练模型权重
├── datasets/              # 数据集目录
│   ├── houston2013/
│   │   ├── HSI_data.mat
│   │   ├── LiDAR_data.mat
│   │   └── All_Label.mat
│   ├── muufl/
│   ├── trento/
│   └── augsburg/
├── log/                   # 模型保存目录
    ├── houston2013.pkl
    ├── muufl.pkl
    └── ...

```


## 📄 许可证

本项目仅供学术研究使用。如需商业使用，请联系作者。

## 🙏 致谢

- 感谢 [CLIP](https://github.com/openai/CLIP) 模型提供的文本-图像对齐能力
- 感谢各数据集提供者的贡献
- 感谢PyTorch社区的支持



## 🔗 相关资源

- 论文：`TGCMFNet/TGCMFNet.pdf`
- 数据集下载：请参考各数据集的官方网站
- CLIP模型：https://github.com/openai/CLIP
