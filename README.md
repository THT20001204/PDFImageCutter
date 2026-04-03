# PDFImageCutter

从 PDF 文件中批量提取嵌入图片，自动筛选质量合格的图片，并将组合图（多子图拼图）智能拆分为独立子图。

## 功能特性

- **嵌入图片提取**：从 PDF 逐页提取所有嵌入的图片对象（非页面截图）
- **质量筛选**：按尺寸、面积、宽高比、文件大小、百万像素等条件自动分类为合格 / 不合格
- **智能子图拆分**：对合格图片自动检测多子图布局，支持轮廓检测、投影分析、网格分割三种策略级联
- **批量处理**：支持文件夹递归扫描，多线程 / 多进程并行
- **元数据记录**：自动生成 CSV + TXT 提取和拆分报告

## 安装

```bash
pip install -r requirements.txt
```

依赖：PyMuPDF、opencv-python、numpy、pandas、Pillow、tqdm

## 使用方法

### 命令行

```bash
# 处理单个 PDF
python PDFimgct.py /path/to/file.pdf -o output_folder

# 处理整个文件夹
python PDFimgct.py /path/to/pdf_folder -o output_folder

# 自定义筛选参数
python PDFimgct.py /path/to/pdf_folder -o output_folder \
    --min-width 100 --min-height 100 --min-area 10000 \
    --min-file-size 5120 --min-megapixels 0.1 --workers 8
```

### 在 Python 中调用

```python
from PDFimgct import simple_complete_process

simple_complete_process(
    pdf_folder="path/to/pdfs",
    output_base_folder="output",
    min_width=50,
    min_height=50,
    min_area=100,
    max_workers=4,
)
```

或使用类接口进行更精细的控制：

```python
from PDFimgct import PDFImageProcessor

processor = PDFImageProcessor(
    base_output_folder="output",
    min_width=100,
    min_height=100,
)

# 处理单个 PDF
result = processor.process_pdf_complete("path/to/file.pdf")

# 批量处理
results = processor.batch_process_pdfs(["path/to/folder"], max_workers=4)
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pdf_path` | （必填） | PDF 文件或文件夹路径 |
| `-o, --output` | `processed_pdf_images` | 输出文件夹 |
| `--min-width` | `50` | 最小宽度（像素） |
| `--min-height` | `50` | 最小高度（像素） |
| `--min-area` | `100` | 最小面积（像素²） |
| `--max-aspect-ratio` | `10` | 最大宽高比 |
| `--min-file-size` | `5120` | 最小文件大小（字节，5KB） |
| `--max-file-size` | `104857600` | 最大文件大小（字节，100MB） |
| `--min-megapixels` | `0.05` | 最小百万像素分辨率 |
| `--workers` | `4` | 并行工作线程数 |

## 输出目录结构

```text
output_folder/
├── pdfs/
│   └── <PDF文件名>/
│       ├── extracted_images/    # 提取的原始图片（临时）
│       ├── qualified_images/    # 合格图片
│       ├── unqualified_images/  # 不合格图片
│       ├── split_images/        # 拆分后的子图
│       ├── failed_splits/       # 拆分失败时保存的原图
│       └── metadata/            # CSV + TXT 元数据报告
├── logs/                        # 处理日志
└── complete_processing_summary.txt  # 汇总报告
```

## 子图拆分策略

对每张合格图片依次尝试：

1. **轮廓检测**：Canny / 自适应阈值 / OTSU，选取检测到最多有效轮廓的方法
2. **投影分析**：水平 + 垂直投影寻找行列分隔线
3. **固定网格**：默认 2×4 网格均分
4. **保存原图**：以上全部失败时保留原图并记录到 `failed_splits/`
