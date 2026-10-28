# GreenTea-8 Dataset Instructions

## Dataset Structure

Place your dataset images in the following directory structure:

```
dataset/data/
├── train/
│   ├── AJBC/        (213 images — Anji Baicha)
│   ├── BLC/         (214 images — Biluochun)
│   ├── ESYL/        (213 images — Enshi Yulu)
│   ├── HSMF/        (214 images — Huangshan Maofeng)
│   ├── LSYW/        (213 images — Lushan Yunwu)
│   ├── LJ/          (214 images — Longjing)
│   └── XYMJ_ZYQ/    (427 images — Xinyang Maojian + Zhuyeqing)
└── val/
    ├── AJBC/        (37 images)
    ├── BLC/         (37 images)
    ├── ESYL/        (37 images)
    ├── HSMF/        (37 images)
    ├── LSYW/        (37 images)
    ├── LJ/          (37 images)
    └── XYMJ_ZYQ/    (74 images)
```

## Getting the Dataset

### Option 1: Download from official repository
```bash
bash scripts/download_dataset.sh
```

### Option 2: Manual placement
1. Create the directory structure above
2. Place images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`) in their respective class folders
3. Verify counts match the expected numbers

## Class Descriptions

| Code      | English Name                          | Chinese Name         | Train | Val |
|-----------|---------------------------------------|---------------------|-------|-----|
| AJBC      | Anji Baicha                           | 安吉白茶              | 213   | 37  |
| BLC       | Biluochun                             | 碧螺春                | 214   | 37  |
| ESYL      | Enshi Yulu                            | 恩施玉露              | 213   | 37  |
| HSMF      | Huangshan Maofeng                     | 黄山毛峰              | 214   | 37  |
| LSYW      | Lushan Yunwu                          | 庐山云雾              | 213   | 37  |
| LJ        | Longjing (Dragon Well)                | 龙井                  | 214   | 37  |
| XYMJ_ZYQ  | Xinyang Maojian + Zhuyeqing (merged)  | 信阳毛尖+竹叶青       | 427   | 74  |

## Notes

- **XYMJ+ZYQ merged**: Due to high visual similarity (both needle-shaped green teas), Xinyang Maojian and Zhuyeqing are merged into one category.
- **Image sources**: Images were collected from publicly accessible e-commerce platforms (Taobao, JD.com) and lifestyle blogs.
- **License**: CC BY-NC 4.0 (for non-commercial academic research only).
