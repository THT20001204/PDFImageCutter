import os
import time
import logging
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import shutil


class PDFImageProcessor:
    """
    PDF图片处理系统 - 提取PDF嵌入图片并自动拆分子图
    """

    def __init__(self, base_output_folder="processed_pdf_images",
                 min_width=100, min_height=100, min_area=10000, max_aspect_ratio=10,
                 min_file_size=1024, max_file_size=50 * 1024 * 1024, min_megapixels=0.1):
        """
        初始化处理器

        Args:
            base_output_folder: 基础输出文件夹
            min_width: 最小宽度（像素）
            min_height: 最小高度（像素）
            min_area: 最小面积（像素²）
            max_aspect_ratio: 最大宽高比
            min_file_size: 最小文件大小（字节，默认1KB）
            max_file_size: 最大文件大小（字节，默认50MB）
            min_megapixels: 最小百万像素分辨率（默认0.1MP，即10万像素）
        """
        self.base_output_folder = base_output_folder
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area
        self.max_aspect_ratio = max_aspect_ratio
        self.min_file_size = min_file_size
        self.max_file_size = max_file_size
        self.min_megapixels = min_megapixels
        self.setup_logging()

    def setup_logging(self):
        """设置日志系统"""
        log_folder = os.path.join(self.base_output_folder, "logs")
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_folder, f"processing_{time.strftime('%Y%m%d_%H%M%S')}.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # 记录筛选条件
        self.logger.info(f"图片筛选条件:")
        self.logger.info(f"  尺寸: {self.min_width}x{self.min_height} 像素 (最小面积: {self.min_area} 像素²)")
        self.logger.info(f"  宽高比: ≤ {self.max_aspect_ratio}")
        self.logger.info(f"  文件大小: {self.min_file_size / 1024:.1f}KB - {self.max_file_size / (1024 * 1024):.1f}MB")
        self.logger.info(f"  百万像素分辨率: ≥ {self.min_megapixels}MP")

    def find_pdf_files(self, input_path):
        """
        查找PDF文件

        Args:
            input_path: 可以是文件、文件夹或包含多个文件夹的列表

        Returns:
            list: PDF文件路径列表
        """
        pdf_files = []

        if isinstance(input_path, str):
            input_path = [input_path]

        for path in input_path:
            if os.path.isfile(path) and path.lower().endswith('.pdf'):
                pdf_files.append(path)
                self.logger.info(f"找到PDF文件: {path}")
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith('.pdf'):
                            pdf_path = os.path.join(root, file)
                            pdf_files.append(pdf_path)
                            self.logger.info(f"找到PDF文件: {pdf_path}")

        self.logger.info(f"总共找到 {len(pdf_files)} 个PDF文件")
        return pdf_files

    def create_output_structure(self, pdf_path):
        """
        为PDF文件创建标准化的输出文件夹结构

        Args:
            pdf_path: PDF文件路径

        Returns:
            dict: 包含各种输出路径的字典
        """
        # 获取PDF文件名（不含扩展名），清理非法字符
        pdf_name = self.sanitize_filename(os.path.splitext(os.path.basename(pdf_path))[0])

        # 创建标准化的文件夹结构
        folders = {
            'pdf_root': os.path.join(self.base_output_folder, "pdfs", pdf_name),
            'extracted_images': os.path.join(self.base_output_folder, "pdfs", pdf_name, "extracted_images"),
            'qualified_images': os.path.join(self.base_output_folder, "pdfs", pdf_name, "qualified_images"),
            'unqualified_images': os.path.join(self.base_output_folder, "pdfs", pdf_name, "unqualified_images"),
            'split_images': os.path.join(self.base_output_folder, "pdfs", pdf_name, "split_images"),
            'metadata': os.path.join(self.base_output_folder, "pdfs", pdf_name, "metadata"),
            'failed_splits': os.path.join(self.base_output_folder, "pdfs", pdf_name, "failed_splits")
        }

        for folder_path in folders.values():
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        return folders

    def sanitize_filename(self, filename):
        """清理文件名中的非法字符"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename

    def calculate_megapixels(self, width, height):
        """
        计算百万像素分辨率

        Args:
            width: 图片宽度（像素）
            height: 图片高度（像素）

        Returns:
            float: 百万像素值
        """
        return (width * height) / 100000.0

    def is_image_qualified(self, width, height, file_size):
        """
        检查图片是否符合所有筛选条件

        Args:
            width: 图片宽度
            height: 图片高度
            file_size: 文件大小

        Returns:
            tuple: (是否符合条件, 不合格原因列表)
        """
        reasons = []

        # 检查尺寸
        if width < self.min_width:
            reasons.append(f"宽度不足 ({width} < {self.min_width})")
        if height < self.min_height:
            reasons.append(f"高度不足 ({height} < {self.min_height})")

        # 检查面积
        area = width * height
        if area < self.min_area:
            reasons.append(f"面积不足 ({area} < {self.min_area})")

        # 检查宽高比
        if min(width, height) > 0:  # 避免除零错误
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > self.max_aspect_ratio:
                reasons.append(f"宽高比过大 ({aspect_ratio:.2f} > {self.max_aspect_ratio})")

        # 检查文件大小
        if file_size < self.min_file_size:
            reasons.append(f"文件大小过小 ({file_size / 1024:.1f}KB < {self.min_file_size / 1024:.1f}KB)")
        if file_size > self.max_file_size:
            reasons.append(
                f"文件大小过大 ({file_size / (1024 * 1024):.1f}MB > {self.max_file_size / (1024 * 1024):.1f}MB)")

        # 检查百万像素分辨率
        megapixels = self.calculate_megapixels(width, height)
        if megapixels < self.min_megapixels:
            reasons.append(f"分辨率不足 ({megapixels:.2f}MP < {self.min_megapixels}MP)")

        qualified = len(reasons) == 0
        return qualified, reasons

    def extract_embedded_images_from_pdf(self, pdf_path):
        """
        从PDF中提取嵌入图片（不提取页面图片）并进行筛选

        Args:
            pdf_path: PDF文件路径

        Returns:
            dict: 提取结果信息
        """
        start_time = time.time()
        result = {
            'pdf_path': pdf_path,
            'success': False,
            'total_images': 0,
            'qualified_images': 0,
            'unqualified_images': 0,
            'error': None,
            'processing_time': 0,
            'output_folders': None,
            'qualified_files': [],
            'unqualified_files': []
        }

        try:
            # 创建输出文件夹结构
            folders = self.create_output_structure(pdf_path)
            result['output_folders'] = folders

            # 打开PDF文件
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)

            image_count = 0
            qualified_count = 0
            unqualified_count = 0
            extraction_details = []

            # 只提取嵌入图像，不提取页面图片
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(pdf_document, xref)

                        # 处理CMYK颜色空间
                        if pix.n - pix.alpha == 4:  # CMYK
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        # 确定文件扩展名
                        ext = self.determine_image_extension(img, pix)

                        # 生成标准化的文件名
                        image_filename = f"page_{page_num + 1:04d}_img_{image_count:04d}{ext}"

                        # 临时保存图片以检查文件大小和DPI
                        temp_image_path = os.path.join(folders['extracted_images'], image_filename)

                        # 保存图像
                        if pix.n < 5:
                            pix.save(temp_image_path)
                        else:
                            pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                            pix_rgb.save(temp_image_path)
                            pix_rgb = None

                        # 获取图片信息
                        width, height = img[2], img[3]
                        file_size = os.path.getsize(temp_image_path)

                        # 检查图片是否符合所有条件
                        is_qualified, reasons = self.is_image_qualified(width, height, file_size)

                        # 根据条件决定最终保存路径
                        if is_qualified:
                            final_folder = folders['qualified_images']
                            result['qualified_files'].append(os.path.join(final_folder, image_filename))
                            qualified_count += 1
                            status = "合格"
                        else:
                            final_folder = folders['unqualified_images']
                            result['unqualified_files'].append(os.path.join(final_folder, image_filename))
                            unqualified_count += 1
                            status = f"不合格: {', '.join(reasons)}"

                        # 移动文件到最终位置
                        final_image_path = os.path.join(final_folder, image_filename)
                        shutil.move(temp_image_path, final_image_path)

                        # 计算百万像素
                        megapixels = self.calculate_megapixels(width, height)

                        # 记录提取详情
                        extraction_details.append({
                            'filename': image_filename,
                            'page': page_num + 1,
                            'image_index': img_index,
                            'width': width,
                            'height': height,
                            'file_size': file_size,
                            'megapixels': megapixels,
                            'file_path': final_image_path,
                            'qualified': is_qualified,
                            'status': status,
                            'reasons': ', '.join(reasons) if reasons else ''
                        })

                        image_count += 1
                        pix = None

                        self.logger.debug(
                            f"提取图片: {image_filename} - {width}x{height} - {file_size / 1024:.1f}KB - {megapixels:.2f}MP - {status}")

                    except Exception as e:
                        self.logger.warning(
                            f"提取嵌入图像失败 (PDF: {os.path.basename(pdf_path)}, 页: {page_num + 1}, 图像: {img_index}): {e}")
                        continue

            pdf_document.close()

            # 保存元数据
            if extraction_details:
                self.save_extraction_metadata(pdf_path, extraction_details, folders['metadata'])

            result['success'] = True
            result['total_images'] = image_count
            result['qualified_images'] = qualified_count
            result['unqualified_images'] = unqualified_count
            result['processing_time'] = time.time() - start_time

            self.logger.info(
                f"成功提取嵌入图片: {os.path.basename(pdf_path)} - "
                f"总计 {image_count} 张图像, 合格 {qualified_count} 张, 不合格 {unqualified_count} 张, "
                f"耗时: {result['processing_time']:.2f}秒")

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            self.logger.error(f"处理PDF失败: {pdf_path} - 错误: {e}")

        return result

    def determine_image_extension(self, img_info, pix):
        """确定图像文件扩展名"""
        try:
            if len(img_info) > 1:
                filters = img_info[1]
                if isinstance(filters, str):
                    filters_str = filters.lower()
                    if "jpxdecode" in filters_str or "jpx" in filters_str:
                        return ".jp2"
                    elif "dctdecode" in filters_str or "jpeg" in filters_str:
                        return ".jpg"
                    elif "ccittfaxdecode" in filters_str:
                        return ".tiff"
        except Exception:
            pass

        # 根据pixmap属性判断
        try:
            if pix.n == 1 or pix.n == 2:
                return ".png"
            elif pix.n == 3:
                return ".png"
            elif pix.n == 4:
                if pix.colorspace and pix.colorspace.n == 4:
                    return ".jpg"
                else:
                    return ".png"
        except Exception:
            pass

        return ".png"

    def save_extraction_metadata(self, pdf_path, extraction_details, metadata_folder):
        """保存提取元数据"""
        pdf_name = self.sanitize_filename(os.path.splitext(os.path.basename(pdf_path))[0])

        # 保存为CSV
        csv_path = os.path.join(metadata_folder, f"{pdf_name}_extraction_metadata.csv")
        df = pd.DataFrame(extraction_details)
        df.to_csv(csv_path, index=False, encoding='utf-8')

        # 保存为文本文件
        txt_path = os.path.join(metadata_folder, f"{pdf_name}_extraction_metadata.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"PDF文件: {os.path.basename(pdf_path)}\n")
            f.write(f"提取时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总嵌入图像数: {len(extraction_details)}\n")

            # 统计合格和不合格的数量
            qualified = [d for d in extraction_details if d['qualified']]
            unqualified = [d for d in extraction_details if not d['qualified']]

            f.write(f"合格图像: {len(qualified)}\n")
            f.write(f"不合格图像: {len(unqualified)}\n")
            f.write(f"筛选条件:\n")
            f.write(f"  最小尺寸: {self.min_width}x{self.min_height} 像素\n")
            f.write(f"  最小面积: {self.min_area} 像素²\n")
            f.write(f"  最大宽高比: {self.max_aspect_ratio}\n")
            f.write(f"  文件大小: {self.min_file_size / 1024:.1f}KB - {self.max_file_size / (1024 * 1024):.1f}MB\n")
            f.write(f"  最小百万像素: {self.min_megapixels}MP\n\n")

            f.write("合格图像详情:\n")
            for detail in qualified:
                f.write(f"图像: {detail['filename']}\n")
                f.write(f"  页面: {detail['page']}\n")
                f.write(f"  图像索引: {detail['image_index']}\n")
                f.write(f"  尺寸: {detail['width']}x{detail['height']}\n")
                f.write(f"  文件大小: {detail['file_size'] / 1024:.1f}KB\n")
                f.write(f"  百万像素: {detail['megapixels']:.2f}MP\n")
                f.write(f"  文件路径: {detail['file_path']}\n\n")

            if unqualified:
                f.write("不合格图像详情:\n")
                for detail in unqualified:
                    f.write(f"图像: {detail['filename']}\n")
                    f.write(f"  页面: {detail['page']}\n")
                    f.write(f"  图像索引: {detail['image_index']}\n")
                    f.write(f"  尺寸: {detail['width']}x{detail['height']}\n")
                    f.write(f"  文件大小: {detail['file_size'] / 1024:.1f}KB\n")
                    f.write(f"  百万像素: {detail['megapixels']:.2f}MP\n")
                    f.write(f"  文件路径: {detail['file_path']}\n")
                    f.write(f"  不合格原因: {detail['reasons']}\n\n")

    def auto_detect_subimages(self, image_path, output_dir):
        """自动检测子图的数量和布局"""
        try:
            # 使用PIL读取并确保RGB格式
            pil_img = Image.open(image_path).convert('RGB')
            img_array = np.array(pil_img)

            # 转换为BGR格式给OpenCV使用
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 尝试多种边缘检测方法
            methods = [
                ('canny', cv2.Canny(gray, 50, 180)),
                ('adaptive_thresh', cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                          cv2.THRESH_BINARY_INV, 11, 2)),
                ('otsu_thresh', cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
            ]

            best_contours = []
            best_method = ""

            for method_name, edges in methods:
                # 形态学操作增强边缘
                kernel = np.ones((3, 3), np.uint8)
                edges_cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

                # 查找轮廓
                contours, _ = cv2.findContours(edges_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 过滤合理的轮廓
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    x, y, w, h = cv2.boundingRect(contour)

                    img_area = img.shape[0] * img.shape[1]
                    if (area > img_area * 0.0002 and
                            area < img_area * 0.8 and
                            w > img.shape[1] * 0.03 and
                            h > img.shape[0] * 0.03 and
                            w / h > 0.3 and w / h < 10):
                        valid_contours.append(contour)

                if len(valid_contours) > len(best_contours):
                    best_contours = valid_contours
                    best_method = method_name

            # 如果自动检测失败，使用投影分析
            if len(best_contours) < 2:
                return self.detect_grid_layout(image_path, output_dir)

            # 计算所有子图的统计信息
            widths = []
            heights = []
            areas = []
            for contour in best_contours:
                x, y, w, h = cv2.boundingRect(contour)
                widths.append(w)
                heights.append(h)
                areas.append(w * h)

            median_width = np.median(widths) if widths else 0
            median_height = np.median(heights) if heights else 0
            median_area = np.median(areas) if areas else 0

            # 根据轮廓提取子图
            success_count = 0
            split_details = []

            for i, contour in enumerate(best_contours):
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                img_area = img.shape[0] * img.shape[1]
                if (area < median_area * 0.25 or
                        (w < median_width * 0.71 and h < median_height * 0.75) or
                        area < img_area * 0.001):
                    continue

                # 添加边距
                margin = 1
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(img.shape[1] - x, w + 2 * margin)
                h = min(img.shape[0] - y, h + 2 * margin)

                # 提取子图
                sub_img = img[y:y + h, x:x + w]

                if sub_img.size > 0:
                    # 标准化的子图文件名
                    subimage_filename = f"subimage_{success_count + 1:04d}.png"
                    output_path = os.path.join(output_dir, subimage_filename)

                    # 使用PIL保存
                    sub_img_rgb = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
                    pil_sub_img = Image.fromarray(sub_img_rgb)
                    pil_sub_img.save(output_path, "PNG")

                    # 计算百万像素
                    megapixels = self.calculate_megapixels(w, h)

                    split_details.append({
                        'filename': subimage_filename,
                        'original_image': os.path.basename(image_path),
                        'position': f"({x}, {y})",
                        'width': w,
                        'height': h,
                        'file_size': os.path.getsize(output_path),
                        'megapixels': megapixels,
                        'file_path': output_path,
                        'split_status': 'success'
                    })

                    success_count += 1

            return success_count, split_details

        except Exception as e:
            self.logger.error(f"自动检测子图失败: {image_path} - 错误: {e}")
            return 0, []

    def detect_grid_layout(self, image_path, output_dir):
        """通过投影分析检测网格布局"""
        try:
            pil_img = Image.open(image_path).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            os.makedirs(output_dir, exist_ok=True)

            # 二值化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 水平投影和垂直投影
            horizontal_projection = np.sum(binary, axis=1)
            vertical_projection = np.sum(binary, axis=0)

            # 寻找行列分隔
            row_boundaries = self.find_boundaries(horizontal_projection, min_gap=img.shape[0] // 20)
            col_boundaries = self.find_boundaries(vertical_projection, min_gap=img.shape[1] // 20)

            # 至少要有2行2列
            if len(row_boundaries) < 3:
                row_boundaries = [0, img.shape[0] // 2, img.shape[0]]
            if len(col_boundaries) < 3:
                col_boundaries = [0, img.shape[1] // 2, img.shape[1]]

            # 计算网格单元统计信息
            cell_widths = []
            cell_heights = []
            cell_areas = []
            for i in range(len(row_boundaries) - 1):
                for j in range(len(col_boundaries) - 1):
                    y1, y2 = row_boundaries[i], row_boundaries[i + 1]
                    x1, x2 = col_boundaries[j], col_boundaries[j + 1]
                    cell_widths.append(x2 - x1)
                    cell_heights.append(y2 - y1)
                    cell_areas.append((x2 - x1) * (y2 - y1))

            median_cell_width = np.median(cell_widths) if cell_widths else 0
            median_cell_height = np.median(cell_heights) if cell_heights else 0
            median_cell_area = np.median(cell_areas) if cell_areas else 0

            # 根据检测到的行列进行分割
            success_count = 0
            split_details = []

            for i in range(len(row_boundaries) - 1):
                for j in range(len(col_boundaries) - 1):
                    y1, y2 = row_boundaries[i], row_boundaries[i + 1]
                    x1, x2 = col_boundaries[j], col_boundaries[j + 1]

                    cell_width = x2 - x1
                    cell_height = y2 - y1
                    cell_area = cell_width * cell_height

                    # 筛选条件
                    if (cell_area < median_cell_area * 0.25 or
                            (cell_width < median_cell_width * 0.5 and cell_height < median_cell_height * 0.5) or
                            cell_width < 100 or cell_height < 100):
                        continue

                    sub_img = img[y1:y2, x1:x2]

                    if sub_img.size > 0:
                        # 标准化的网格子图文件名
                        subimage_filename = f"grid_{i}_{j}.png"
                        output_path = os.path.join(output_dir, subimage_filename)

                        sub_img_rgb = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
                        pil_sub_img = Image.fromarray(sub_img_rgb)
                        pil_sub_img.save(output_path, "PNG")

                        # 计算百万像素
                        megapixels = self.calculate_megapixels(cell_width, cell_height)

                        split_details.append({
                            'filename': subimage_filename,
                            'original_image': os.path.basename(image_path),
                            'grid_position': f"({i}, {j})",
                            'width': cell_width,
                            'height': cell_height,
                            'file_size': os.path.getsize(output_path),
                            'megapixels': megapixels,
                            'file_path': output_path,
                            'split_status': 'success'
                        })

                        success_count += 1

            return success_count, split_details

        except Exception as e:
            self.logger.error(f"网格布局检测失败: {image_path} - 错误: {e}")
            return 0, []

    def find_boundaries(self, projection, min_gap=20):

        valleys = []
        in_valley = False
        valley_start = 0

        threshold = np.max(projection) * 0.1

        for i, value in enumerate(projection):
            if value < threshold and not in_valley:
                in_valley = True
                valley_start = i
            elif value >= threshold and in_valley:
                in_valley = False
                if i - valley_start >= min_gap:
                    valleys.append((valley_start, i))

        if len(valleys) < 1:
            length = len(projection)
            return [0, length // 2, length]

        boundaries = [0]
        for start, end in valleys:
            boundaries.append((start + end) // 2)
        boundaries.append(len(projection))

        return boundaries

    def robust_auto_split(self, image_path, output_dir, failed_splits_dir):
        """
        健壮的自动分割方法 - 结合多种检测方法
        如果拆分失败，保存原图到失败目录
        """
        # 方法1: 自动轮廓检测
        count1, details1 = self.auto_detect_subimages(image_path, output_dir + "_contour")

        # 方法2: 投影分析
        count2, details2 = self.detect_grid_layout(image_path, output_dir + "_projection")

        # 选择最佳结果
        if count1 >= count2 and count1 > 0:
            # 复制最佳结果到主输出目录
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            shutil.copytree(output_dir + "_contour", output_dir)
            return count1, details1
        elif count2 > 0:
            # 复制最佳结果到主输出目录
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            shutil.copytree(output_dir + "_projection", output_dir)
            return count2, details2
        else:
            # 使用默认2x4网格分割
            count3, details3 = self.manual_grid_split(image_path, output_dir, 2, 4)
            if count3 > 0:
                return count3, details3
            else:
                # 所有方法都失败，保存原图到失败目录
                return self.save_original_as_fallback(image_path, output_dir, failed_splits_dir)

    def manual_grid_split(self, image_path, output_dir, rows, cols):
        """手动指定行列数的网格分割"""
        try:
            pil_img = Image.open(image_path).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            os.makedirs(output_dir, exist_ok=True)

            h, w = img.shape[:2]
            cell_width = w // cols
            cell_height = h // rows
            cell_area = cell_width * cell_height

            success_count = 0
            split_details = []

            for i in range(rows):
                for j in range(cols):
                    x1 = j * w // cols
                    x2 = (j + 1) * w // cols
                    y1 = i * h // rows
                    y2 = (i + 1) * h // rows

                    cell_width_actual = x2 - x1
                    cell_height_actual = y2 - y1
                    cell_area_actual = cell_width_actual * cell_height_actual

                    if (cell_area_actual < cell_area * 0.25 or
                            (cell_width_actual < cell_width * 0.5 and cell_height_actual < cell_height * 0.5) or
                            cell_width_actual < 100 or cell_height_actual < 100):
                        continue

                    sub_img = img[y1:y2, x1:x2]

                    if sub_img.size > 0:

                        subimage_filename = f"manual_{i}_{j}.png"
                        output_path = os.path.join(output_dir, subimage_filename)

                        sub_img_rgb = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
                        pil_sub_img = Image.fromarray(sub_img_rgb)
                        pil_sub_img.save(output_path, "PNG")

                        # 计算分辨率
                        megapixels = self.calculate_megapixels(cell_width_actual, cell_height_actual)

                        split_details.append({
                            'filename': subimage_filename,
                            'original_image': os.path.basename(image_path),
                            'grid_position': f"({i}, {j})",
                            'width': cell_width_actual,
                            'height': cell_height_actual,
                            'file_size': os.path.getsize(output_path),
                            'megapixels': megapixels,
                            'file_path': output_path,
                            'split_status': 'success'
                        })

                        success_count += 1

            return success_count, split_details

        except Exception as e:
            self.logger.error(f"手动网格分割失败: {image_path} - 错误: {e}")
            return 0, []

    def save_original_as_fallback(self, image_path, output_dir, failed_splits_dir):
        """
        拆分失败时保存原图

        Args:
            image_path: 原图路径
            output_dir: 拆分输出目录
            failed_splits_dir: 失败拆分目录

        Returns:
            tuple: (图片数量, 详情列表)
        """
        try:

            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(failed_splits_dir, exist_ok=True)

            original_filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(original_filename)[0]

            output_filename = f"{name_without_ext}_original.png"
            output_path = os.path.join(output_dir, output_filename)

            failed_output_path = os.path.join(failed_splits_dir, output_filename)

            pil_img = Image.open(image_path).convert('RGB')
            width, height = pil_img.size
            pil_img.save(output_path, "PNG")
            pil_img.save(failed_output_path, "PNG")

            # 获取图片信息
            file_size = os.path.getsize(output_path)
            megapixels = self.calculate_megapixels(width, height)

            detail = {
                'filename': output_filename,
                'original_image': original_filename,
                'width': width,
                'height': height,
                'file_size': file_size,
                'megapixels': megapixels,
                'file_path': output_path,
                'split_status': 'failed',
                'reason': '所有拆分方法都失败，保存原图'
            }

            self.logger.warning(f"拆分失败，保存原图: {original_filename} -> {output_filename}")

            return 1, [detail]

        except Exception as e:
            self.logger.error(f"保存原图失败: {image_path} - 错误: {e}")
            return 0, []

    def split_qualified_images(self, extraction_result):
        """
        对合格的图片进行拆分（不合格的图片不处理）

        Args:
            extraction_result: 提取结果信息

        Returns:
            dict: 拆分结果信息
        """
        if not extraction_result['success']:
            return {'success': False, 'error': '提取失败，无法进行拆分'}

        start_time = time.time()
        split_result = {
            'pdf_path': extraction_result['pdf_path'],
            'success': True,
            'total_qualified_images': len(extraction_result['qualified_files']),
            'total_split_images': 0,
            'successful_splits': 0,
            'failed_splits': 0,
            'split_details': [],
            'processing_time': 0
        }

        try:
            folders = extraction_result['output_folders']
            all_split_details = []

            # 只对合格的图片进行拆分
            for image_path in extraction_result['qualified_files']:
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                split_output_dir = os.path.join(folders['split_images'], image_name)

                # 进行图片拆分
                count, details = self.robust_auto_split(
                    image_path,
                    split_output_dir,
                    folders['failed_splits']
                )

                split_result['total_split_images'] += count

                # 统计成功和失败的数量
                for detail in details:
                    if detail.get('split_status') == 'success':
                        split_result['successful_splits'] += 1
                    else:
                        split_result['failed_splits'] += 1

                all_split_details.extend(details)

                status = "成功" if count > 0 else "失败(保存原图)"
                self.logger.info(f"拆分{status}: {os.path.basename(image_path)} -> {count} 个子图")

            # 保存拆分元数据
            if all_split_details:
                self.save_split_metadata(extraction_result['pdf_path'], all_split_details, folders['metadata'])

            split_result['split_details'] = all_split_details
            split_result['processing_time'] = time.time() - start_time

            self.logger.info(f"拆分完成: {os.path.basename(extraction_result['pdf_path'])} - "
                             f"合格图片: {split_result['total_qualified_images']}, "
                             f"总子图数: {split_result['total_split_images']}, "
                             f"成功拆分: {split_result['successful_splits']}, "
                             f"失败保存原图: {split_result['failed_splits']}, "
                             f"耗时: {split_result['processing_time']:.2f}秒")

        except Exception as e:
            split_result['success'] = False
            split_result['error'] = str(e)
            self.logger.error(f"图片拆分失败: {extraction_result['pdf_path']} - 错误: {e}")

        return split_result

    def save_split_metadata(self, pdf_path, split_details, metadata_folder):
        """保存拆分元数据"""
        pdf_name = self.sanitize_filename(os.path.splitext(os.path.basename(pdf_path))[0])

        # 保存为CSV
        csv_path = os.path.join(metadata_folder, f"{pdf_name}_split_metadata.csv")
        df = pd.DataFrame(split_details)
        df.to_csv(csv_path, index=False, encoding='utf-8')

        # 保存为文本文件
        txt_path = os.path.join(metadata_folder, f"{pdf_name}_split_metadata.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"PDF文件: {os.path.basename(pdf_path)}\n")
            f.write(f"拆分时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总子图数: {len(split_details)}\n")

            # 统计成功和失败的数量
            successful = [d for d in split_details if d.get('split_status') == 'success']
            failed = [d for d in split_details if d.get('split_status') == 'failed']

            f.write(f"成功拆分子图: {len(successful)}\n")
            f.write(f"失败保存原图: {len(failed)}\n\n")

            f.write("成功拆分子图详情:\n")
            for detail in successful:
                f.write(f"子图: {detail['filename']}\n")
                f.write(f"  原图像: {detail['original_image']}\n")
                if 'position' in detail:
                    f.write(f"  位置: {detail['position']}\n")
                if 'grid_position' in detail:
                    f.write(f"  网格位置: {detail['grid_position']}\n")
                f.write(f"  尺寸: {detail['width']}x{detail['height']}\n")
                f.write(f"  文件大小: {detail['file_size'] / 1024:.1f}KB\n")
                f.write(f"  百万像素: {detail['megapixels']:.2f}MP\n")
                f.write(f"  文件路径: {detail['file_path']}\n\n")

            if failed:
                f.write("失败保存原图详情:\n")
                for detail in failed:
                    f.write(f"原图: {detail['filename']}\n")
                    f.write(f"  原图像: {detail['original_image']}\n")
                    f.write(f"  尺寸: {detail['width']}x{detail['height']}\n")
                    f.write(f"  文件大小: {detail['file_size'] / 1024:.1f}KB\n")
                    f.write(f"  百万像素: {detail['megapixels']:.2f}MP\n")
                    f.write(f"  文件路径: {detail['file_path']}\n")
                    f.write(f"  失败原因: {detail.get('reason', '未知')}\n\n")

    def process_pdf_complete(self, pdf_path):
        """
        完整处理单个PDF文件：提取嵌入图片 + 筛选 + 对合格图片拆分子图

        Args:
            pdf_path: PDF文件路径

        Returns:
            dict: 完整处理结果
        """
        self.logger.info(f"开始完整处理PDF: {pdf_path}")

        extraction_result = self.extract_embedded_images_from_pdf(pdf_path)

        if not extraction_result['success']:
            return {
                'pdf_path': pdf_path,
                'success': False,
                'error': f"嵌入图片提取失败: {extraction_result['error']}",
                'extraction_result': extraction_result,
                'split_result': None
            }

        # 步骤2: 只对合格的图片进行拆分
        split_result = self.split_qualified_images(extraction_result)

        complete_result = {
            'pdf_path': pdf_path,
            'success': split_result['success'],
            'extraction_result': extraction_result,
            'split_result': split_result
        }

        if split_result['success']:
            self.logger.info(f"PDF处理完成: {os.path.basename(pdf_path)} - "
                             f"提取 {extraction_result['total_images']} 张嵌入图片 "
                             f"(合格: {extraction_result['qualified_images']}, 不合格: {extraction_result['unqualified_images']}), "
                             f"拆分 {split_result['total_split_images']} 个子图 "
                             f"(成功: {split_result['successful_splits']}, 失败: {split_result['failed_splits']})")
        else:
            self.logger.warning(f"PDF处理部分完成: {os.path.basename(pdf_path)} - "
                                f"提取成功但拆分失败: {split_result['error']}")

        return complete_result

    def batch_process_pdfs(self, input_paths, max_workers=None, use_multiprocessing=False):
        """
        批量处理PDF文件

        Args:
            input_paths: PDF文件路径或文件夹列表
            max_workers: 最大工作线程数
            use_multiprocessing: 是否使用多进程

        Returns:
            list: 所有PDF的完整处理结果
        """
        # 查找所有PDF文件
        pdf_files = self.find_pdf_files(input_paths)

        if not pdf_files:
            self.logger.warning("未找到任何PDF文件")
            return []

        self.logger.info(f"开始批量处理 {len(pdf_files)} 个PDF文件")

        all_results = []

        # 单线程处理
        if max_workers == 1:
            for pdf_file in tqdm(pdf_files, desc="处理PDF文件"):
                result = self.process_pdf_complete(pdf_file)
                all_results.append(result)

        # 多线程/多进程处理
        else:
            executor_class = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor

            with executor_class(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.process_pdf_complete, pdf_file)
                    for pdf_file in pdf_files
                ]

                for future in tqdm(futures, desc="处理PDF文件", total=len(futures)):
                    try:
                        result = future.result(timeout=600)  # 10分钟超时
                        all_results.append(result)
                    except Exception as e:
                        self.logger.error(f"处理任务失败: {e}")
                        all_results.append({
                            'pdf_path': 'Unknown',
                            'success': False,
                            'error': str(e)
                        })

        # 生成汇总报告
        self.generate_complete_summary_report(all_results)

        return all_results

    def generate_complete_summary_report(self, all_results):
        """生成完整处理汇总报告"""
        successful = [r for r in all_results if r['success']]
        failed = [r for r in all_results if not r['success']]

        total_extracted = sum(r['extraction_result']['total_images'] for r in successful if r['extraction_result'])
        total_qualified = sum(r['extraction_result']['qualified_images'] for r in successful if r['extraction_result'])
        total_unqualified = sum(
            r['extraction_result']['unqualified_images'] for r in successful if r['extraction_result'])
        total_split = sum(r['split_result']['total_split_images'] for r in successful if r['split_result'])
        total_successful_splits = sum(r['split_result']['successful_splits'] for r in successful if r['split_result'])
        total_failed_splits = sum(r['split_result']['failed_splits'] for r in successful if r['split_result'])

        total_time = 0
        for result in successful:
            if result['extraction_result'] and result['split_result']:
                total_time += (result['extraction_result']['processing_time'] +
                               result['split_result']['processing_time'])

        report_path = os.path.join(self.base_output_folder, "complete_processing_summary.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("PDF嵌入图片提取与拆分完整处理汇总报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"筛选条件:\n")
            f.write(f"  最小尺寸: {self.min_width}x{self.min_height} 像素\n")
            f.write(f"  最小面积: {self.min_area} 像素²\n")
            f.write(f"  最大宽高比: {self.max_aspect_ratio}\n")
            f.write(f"  文件大小: {self.min_file_size / 1024:.1f}KB - {self.max_file_size / (1024 * 1024):.1f}MB\n")
            f.write(f"  最小百万像素: {self.min_megapixels}MP\n")
            f.write(f"总PDF文件数: {len(all_results)}\n")
            f.write(f"成功处理: {len(successful)}\n")
            f.write(f"处理失败: {len(failed)}\n")
            f.write(f"总提取嵌入图像数: {total_extracted}\n")
            f.write(f"合格图像: {total_qualified}\n")
            f.write(f"不合格图像: {total_unqualified}\n")
            f.write(f"总拆分子图数: {total_split}\n")
            f.write(f"成功拆分子图: {total_successful_splits}\n")
            f.write(f"失败保存原图: {total_failed_splits}\n")
            f.write(f"总处理时间: {total_time:.2f} 秒\n\n")

            if successful:
                f.write("成功处理文件详情:\n")
                for result in successful:
                    pdf_name = os.path.basename(result['pdf_path'])
                    extracted = result['extraction_result']['total_images']
                    qualified = result['extraction_result']['qualified_images']
                    unqualified = result['extraction_result']['unqualified_images']
                    split = result['split_result']['total_split_images']
                    successful_splits = result['split_result']['successful_splits']
                    failed_splits = result['split_result']['failed_splits']
                    f.write(f"  - {pdf_name}: 提取 {extracted} 张图 (合格: {qualified}, 不合格: {unqualified}), "
                            f"拆分 {split} 个子图 (成功: {successful_splits}, 失败: {failed_splits})\n")

            if failed:
                f.write("\n失败文件列表:\n")
                for result in failed:
                    f.write(f"  - {result['pdf_path']}: {result.get('error', '未知错误')}\n")

        self.logger.info(f"批量处理完成! 汇总报告已保存至: {report_path}")
        self.logger.info(f"成功: {len(successful)}/{len(all_results)}, "
                         f"提取图像: {total_extracted} (合格: {total_qualified}, 不合格: {total_unqualified}), "
                         f"拆分子图: {total_split} (成功: {total_successful_splits}, 失败: {total_failed_splits})")


# 简单使用函数
def simple_complete_process(pdf_folder, output_base_folder="processed_pdf_images",
                            min_width=100, min_height=100, min_area=10000, max_aspect_ratio=10,
                            min_file_size=1024, max_file_size=50 * 1024 * 1024, min_megapixels=0.1,
                            max_workers=4):
    """
    简单完整处理函数

    Args:
        pdf_folder: PDF文件夹路径
        output_base_folder: 输出文件夹
        min_width: 最小宽度（像素）
        min_height: 最小高度（像素）
        min_area: 最小面积（像素²）
        max_aspect_ratio: 最大宽高比
        min_file_size: 最小文件大小（字节，默认1KB）
        max_file_size: 最大文件大小（字节，默认50MB）
        min_megapixels: 最小百万像素分辨率（默认0.1MP）
        max_workers: 并行工作数
    """
    processor = PDFImageProcessor(
        output_base_folder,
        min_width=min_width,
        min_height=min_height,
        min_area=min_area,
        max_aspect_ratio=max_aspect_ratio,
        min_file_size=min_file_size,
        max_file_size=max_file_size,
        min_megapixels=min_megapixels
    )
    processor.batch_process_pdfs(pdf_folder, max_workers=max_workers)


if __name__ == "__main__":
    # 使用示例
    pdf_folder = r"C:\Users\19191\Downloads\刘成下载\刘成下载\上线文章"  # 替换为你的PDF文件夹
    output_folder = r"C:\Users\19191\Desktop\PDF完整处理"

    # 设置筛选条件
    min_width = 50  # 最小宽度
    min_height = 50  # 最小高度
    min_area = 100  # 最小面积
    max_aspect_ratio = 10  # 最大宽高比
    min_file_size = 1024*5  # 最小文件大小 (5KB)
    max_file_size = 100 * 1024 * 1024  # 最大文件大小 (50MB)
    min_megapixels = 0.05  # 最小百万像素分辨率 (0.1MP = 10万像素)

    if os.path.exists(pdf_folder):
        simple_complete_process(
            pdf_folder,
            output_folder,
            min_width=min_width,
            min_height=min_height,
            min_area=min_area,
            max_aspect_ratio=max_aspect_ratio,
            min_file_size=min_file_size,
            max_file_size=max_file_size,
            min_megapixels=min_megapixels,
            max_workers=4
        )
    else:
        print(f"PDF文件夹不存在: {pdf_folder}")
