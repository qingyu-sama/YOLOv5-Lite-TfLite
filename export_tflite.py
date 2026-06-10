#!/usr/bin/env python3
"""
Export YOLOv5-Lite model to TFLite format with full integer quantization.

完整分析 ultralytics TFLite full integer quantization 导出流程，并为 YOLOv5-Lite 实现相同功能。

=============================================================================
Ultralytics TFLite Full Integer Quantization 流程分析:
=============================================================================

1. PyTorch → ONNX (export_onnx)
   - 模型 deepcopy, eval mode, fuse Conv+BN
   - torch.onnx.export 导出为 ONNX, opset 12+
   - 使用 onnxslim 简化 graph
   - 添加 metadata

2. ONNX → TensorFlow SavedModel + TFLite (export_saved_model → onnx2saved_model)
   - 使用 onnx2tf 库将 ONNX 转换为 TensorFlow SavedModel
   - int8=True 时:
     a. 通过 get_int8_calibration_dataloader() 收集标定数据集
     b. 将图像 resize 到模型输入尺寸, 保存为 BHWC 格式的 .npy 文件
     c. 传入 calibration 数据路径到 onnx2tf.convert()
   - onnx2tf.convert() 核心参数:
     * output_integer_quantized_tflite=True   → 生成 full integer quant TFLite
     * custom_input_op_name_np_data_path=...  → 标定数据路径 (格式: [["images", npy_path, [[[[0,0,0]]]], [[[[255,255,255]]]]]])
     * not_use_onnxsim=True                   → 不使用 ONNX simplifier (已提前简化)
     * verbosity="error"                      → 最小日志输出
     * output_signaturedefs=True              → 输出 signature definitions

3. TFLite 文件管理 (onnx2saved_model 内部):
   - *_float32.tflite                        → FP32 模型 (保留)
   - *_dynamic_range_quant.tflite            → 重命名为 *_int8.tflite (动态范围量化)
   - *_integer_quant.tflite / *_full_integer_quant.tflite → Full integer 量化 (INT8 input/output)
   - *_integer_quant_with_int16_act.tflite   → 删除 (额外 FP16 activation 文件)

4. export_tflite 方法:
   - 简单返回已生成的 .tflite 文件路径
   - 实际工作在 export_saved_model 中完成

=============================================================================
关键差异 - YOLOv5-Lite 适配:
=============================================================================
- YOLOv5-Lite 的 Detect 头输出 (predictions, raw_features) 元组, 需要处理为单输出
- 使用 cat_forward 将多尺度检测结果拼接为单一 tensor
- Shuffle_Block 的 channel_shuffle 操作可能被 onnx2tf 拒绝, 需要添加
  disable_group_convolution 参数

Usage:
    # 基础 FP32 导出
    python export_tflite.py --weights v5lite-s.pt --img-size 320 320

    # INT8 动态范围量化 (无需标定数据)
    python export_tflite.py --weights v5lite-s.pt --img-size 320 320 --int8

    # INT8 全整数量化 (需要标定数据, full integer)
    python export_tflite.py --weights v5lite-s.pt --img-size 320 320 --int8 --full-integer --data coco128.yaml

    # FP16 量化
    python export_tflite.py --weights v5lite-s.pt --img-size 320 320 --half

    # 自定义标定图片目录
    python export_tflite.py --weights v5lite-s.pt --img-size 320 320 --int8 --full-integer --calibration-dir /path/to/images

    # 仅导出 ONNX (不进行 TFLite 转换)
    python export_tflite.py --weights v5lite-s.pt --img-size 320 320 --onnx-only

Requirements:
    pip install tensorflow==2.19.0 tf_keras<=2.19.0
    pip install onnx>=1.12.0,<2.0.0 onnxslim>=0.1.71 onnxruntime
    pip install onnx2tf>=1.26.3,<1.29.0
    pip install sng4onnx>=1.0.1 onnx_graphsurgeon>=0.3.26
    pip install ai-edge-litert>=1.2.0
    pip install protobuf>=5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Ensure YOLOv5-Lite root is on path for imports
FILE = Path(__file__).resolve()
ROOT = FILE.parent
sys.path.insert(0, str(ROOT))

import models
from models.common import NMS, NMS_Export
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.datasets import LoadImages, letterbox
from utils.general import check_dataset, check_file, check_img_size, set_logging
from utils.torch_utils import select_device


def parse_args():
    """Parse command-line arguments for TFLite export."""
    parser = argparse.ArgumentParser(
        description="Export YOLOv5-Lite to TFLite with full integer quantization support."
    )
    parser.add_argument(
        "--weights", type=str, default="v5lite-s.pt",
        help="Path to YOLOv5-Lite weights (.pt file)."
    )
    parser.add_argument(
        "--img-size", nargs="+", type=int, default=[320, 320],
        help="Model input image size (height, width)."
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export.")
    parser.add_argument(
        "--int8", action="store_true",
        help="Enable INT8 quantization (dynamic range). Use --full-integer for full integer quantization."
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Enable FP16 quantization (mutually exclusive with --int8)."
    )
    parser.add_argument(
        "--full-integer", action="store_true",
        help="Enable full integer quantization (INT8 input + INT8 output). Requires --int8."
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to dataset YAML for INT8 calibration (e.g., coco128.yaml)."
    )
    parser.add_argument(
        "--calibration-dir", type=str, default=None,
        help="Path to directory of images for INT8 calibration (alternative to --data)."
    )
    parser.add_argument(
        "--num-calibration-images", type=int, default=20,
        help="Number of images to use for INT8 calibration (default: 20, recommended: 20-100)."
    )
    parser.add_argument(
        "--dynamic", action="store_true",
        help="Enable dynamic axes in ONNX export (batch, height, width)."
    )
    parser.add_argument(
        "--opset", type=int, default=18,
        help="ONNX opset version (default: 18, minimum 12). Note: PyTorch 2.12+ requires opset >= 18."
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for model export (cpu or cuda:0)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for exported models (default: same directory as weights)."
    )
    parser.add_argument(
        "--onnx-only", action="store_true",
        help="Only export ONNX, skip TFLite conversion."
    )
    parser.add_argument(
        "--no-onnxsim", action="store_true",
        help="Skip ONNX simplification with onnxslim."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging from onnx2tf."
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Custom model name prefix for output files."
    )
    args = parser.parse_args()

    # Validate arguments
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand (h, w)
    if args.half and args.int8:
        parser.error("--half and --int8 are mutually exclusive.")
    if args.full_integer and not args.int8:
        parser.error("--full-integer requires --int8.")
    return args


class TFLiteExporter:
    """
    Export YOLOv5-Lite models to TensorFlow Lite format.

    This class mirrors the ultralytics Exporter class TFLite export flow,
    adapted for the YOLOv5-Lite model architecture.

    Full Integer Quantization Flow (完整流程):
    ==========================================

    PyTorch Model (.pt)
        │
        ├─ 1. Fuse Conv+BatchNorm layers
        │      model.fuse()
        │
        ├─ 2. Replace activations for ONNX compatibility
        │      Hardswish → Hardswish (custom), SiLU → SiLU (custom)
        │
        ├─ 3. Set Detect layer export mode
        │      m.export = True  (disable grid creation during forward)
        │
        ├─ 4. Dry run inference
        │      model(im)  → validates model works
        │
        ├─ 5. ONNX Export
        │      torch.onnx.export(model, im, f_onnx, opset_version=12, ...)
        │      │
        │      ├─ onnxslim.slim()  → 简化 ONNX graph
        │      └─ Add metadata  → description, author, stride, task, names, etc.
        │
        ├─ 6. INT8 Calibration Data Collection (if --int8)
        │      │
        │      ├─ Load images from dataset or directory
        │      ├─ Resize to model input size (LetterBox padding)
        │      ├─ Convert to BHWC float32 numpy array
        │      └─ Save as .npy file
        │
        ├─ 7. ONNX → TFLite via onnx2tf
        │      onnx2tf.convert(
        │          input_onnx_file_path=f_onnx,
        │          output_folder_path=output_dir,
        │          output_integer_quantized_tflite=bool(int8),
        │          custom_input_op_name_np_data_path=calibration_data,
        │          not_use_onnxsim=True,
        │          verbosity="error" or "debug",
        │          output_signaturedefs=True,
        │          disable_group_convolution=True,  # for Shuffle_Block compatibility
        │      )
        │      │
        │      └─ 生成文件:
        │          ├─ model_float32.tflite         FP32 (无量化)
        │          ├─ model_dynamic_range_quant.tflite → 重命名 model_int8.tflite
        │          ├─ model_full_integer_quant.tflite   INT8 输入+输出 (full integer)
        │          └─ model_integer_quant_with_int16_act.tflite → 删除
        │
        └─ 8. Add metadata to TFLite files
               zipfile 写入 metadata.json 到 .tflite 文件
    """

    def __init__(self, args):
        self.args = args
        self.device = select_device(args.device)
        self.weights_path = Path(args.weights)
        self.model_name = args.model_name or self.weights_path.stem
        self.output_dir = Path(args.output_dir) if args.output_dir else self.weights_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model attributes (populated during export)
        self.model = None
        self.im = None
        self.imgsz = None
        self.stride = None
        self.names = None
        self.nc = None

        # Output paths
        self.f_onnx = self.output_dir / f"{self.model_name}.onnx"
        self.saved_model_dir = self.output_dir / f"{self.model_name}_saved_model"

    def load_model(self):
        """Load and prepare the YOLOv5-Lite PyTorch model.

        Steps:
        1. Load weights with PyTorch 2.6+ compatibility
        2. Set model to eval mode, disable gradient
        3. Verify image size against model stride
        """
        print(f"\n{'='*60}")
        print(f"Loading YOLOv5-Lite model from: {self.weights_path}")
        print(f"{'='*60}")

        # ---- PyTorch 2.6+ compatibility ----
        # torch.load defaults to weights_only=True since PyTorch 2.6.
        # YOLOv5 checkpoints use numpy scalars which are not allowlisted by default.
        # Monkey-patch torch.load to force weights_only=False for checkpoint loading.
        _original_torch_load = torch.load
        def _patched_load(*a, **kw):
            kw.setdefault("weights_only", False)
            return _original_torch_load(*a, **kw)
        torch.load = _patched_load
        try:
            # attempt_load handles ensemble of models, loads EMA weights, and calls model.fuse()
            self.model = attempt_load(self.weights_path, map_location=self.device)
        finally:
            torch.load = _original_torch_load

        # Verify the model was loaded successfully
        assert self.model is not None, f"Failed to load model from {self.weights_path}"

        self.model.eval()
        self.model.float()

        # Disable parameter gradients
        for p in self.model.parameters():
            p.requires_grad = False

        # Get model attributes
        self.stride = int(self.model.stride.max()) if hasattr(self.model, 'stride') else 32
        self.names = self.model.names if hasattr(self.model, 'names') else {}
        self.nc = self.model.model[-1].nc if hasattr(self.model.model[-1], 'nc') else 80

        # Validate image size
        self.imgsz = [check_img_size(x, self.stride) for x in self.args.img_size]
        print(f"  Model stride: {self.stride}")
        print(f"  Input size: {self.imgsz}")
        print(f"  Number of classes: {self.nc}")

    def prepare_for_export(self):
        """Prepare model modules for ONNX export.

        - Replace PyTorch activations with export-friendly versions
        - Configure Detect layer for export mode
        - Set Detect head to use cat_forward for single-output tensor
        """
        print(f"\n{'='*60}")
        print(f"Preparing model for ONNX export...")
        print(f"{'='*60}")

        for k, m in self.model.named_modules():
            m._non_persistent_buffers_set = set()  # PyTorch 1.6 compatibility

            if isinstance(m, models.common.Conv):
                # Replace activations with export-friendly versions
                if isinstance(m.act, nn.Hardswish):
                    m.act = Hardswish()
                elif isinstance(m.act, nn.SiLU):
                    m.act = SiLU()

            elif isinstance(m, models.yolo.Detect):
                # Use cat_forward to return single concatenated tensor
                # instead of (predictions, features) tuple
                m.forward = m.cat_forward
                print(f"  Detect layer: using cat_forward (single output tensor)")

            elif isinstance(m, models.common.Shuffle_Block):
                # Use export_forward to avoid channel_shuffle Transpose ops.
                # Replaces concat+channel_shuffle (→ ONNX Reshape+Transpose+Reshape,
                # → TFLite ~5 TRANSPOSE per block) with stack+flatten
                # (→ ONNX Concat+Reshape, → TFLite CONCATENATION+RESHAPE, 0 TRANSPOSE).
                m.forward = m.export_forward
                print(f"  Shuffle_Block (stride={m.stride}): using export_forward (stack+flatten, no Transpose)")

        # Disable grid export for cleaner ONNX graph
        # In cat_forward, grid is still computed but stays local
        self.model.model[-1].export = True
        print(f"  Detect.export = True (dynamic grid computation)")

    def export_onnx(self):
        """Export prepared model to ONNX format.

        This mirrors ultralytics' export_onnx() flow:
        1. Create dummy input tensor
        2. Dry run inference
        3. Export with torch.onnx.export
        4. Simplify with onnxslim
        5. Add metadata
        """
        print(f"\n{'='*60}")
        print(f"Exporting ONNX model...")
        print(f"{'='*60}")

        # Create dummy input
        self.im = torch.zeros(
            self.args.batch_size, 3, *self.imgsz
        ).to(self.device)

        # Dry run inference
        print(f"  Input shape: {tuple(self.im.shape)}")
        with torch.no_grad():
            y = self.model(self.im)
        print(f"  Output type: {type(y)}, shape: {y.shape if isinstance(y, torch.Tensor) else 'N/A'}")

        # ONNX export
        try:
            import onnx
            print(f"  ONNX version: {onnx.__version__}")
        except ImportError:
            raise ImportError("onnx is required. Install with: pip install onnx>=1.12.0,<2.0.0")

        output_names = ["output0"]
        dynamic = None
        if self.args.dynamic:
            dynamic = {
                "images": {0: "batch", 2: "height", 3: "width"},
                "output0": {0: "batch"},
            }

        print(f"  Opset: {self.args.opset}")
        print(f"  Dynamic axes: {dynamic}")
        print(f"  Output names: {output_names}")

        torch.onnx.export(
            self.model,
            self.im,
            str(self.f_onnx),
            verbose=False,
            opset_version=self.args.opset,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=dynamic,
        )

        # Load and verify ONNX model
        onnx_model = onnx.load(str(self.f_onnx))
        onnx.checker.check_model(onnx_model)
        print(f"  ONNX model verified OK")

        # Simplify with onnxslim
        if not self.args.no_onnxsim:
            try:
                import onnxslim
                print(f"  Simplifying with onnxslim {onnxslim.__version__}...")
                onnx_model = onnxslim.slim(onnx_model)
                onnx.save(onnx_model, str(self.f_onnx))
                print(f"  ONNX model simplified OK")
            except ImportError:
                print(f"  WARNING: onnxslim not installed, skipping simplification")
            except Exception as e:
                print(f"  WARNING: ONNX simplification failed: {e}")

        # Add metadata to ONNX model
        metadata = {
            "description": f"YOLOv5-Lite {self.model_name} exported model",
            "author": "YOLOv5-Lite",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stride": self.stride,
            "task": "detect",
            "batch": self.args.batch_size,
            "imgsz": list(self.imgsz),
            "names": {str(i): name for i, name in enumerate(self.names)},
            "nc": self.nc,
        }
        for k, v in metadata.items():
            meta = onnx_model.metadata_props.add()
            meta.key, meta.value = k, str(v)
        onnx.save(onnx_model, str(self.f_onnx))

        self.metadata = metadata
        mb = os.path.getsize(self.f_onnx) / 1e6
        print(f"  ONNX model saved: {self.f_onnx} ({mb:.2f} MB)")
        return str(self.f_onnx)

    def collect_calibration_images(self):
        """Collect calibration images for INT8 quantization.

        This mirrors ultralytics' get_int8_calibration_dataloader() + preprocessing:
        1. Load images from dataset YAML or image directory
        2. Resize to model input size with LetterBox padding (keeping aspect ratio)
        3. Convert to BHWC float32 numpy array in range [0, 255]

        Returns:
            numpy.ndarray: Calibration images in BHWC format, shape (N, H, W, 3), dtype float32.
            None if no calibration source is available.
        """
        print(f"\n{'='*60}")
        print(f"Collecting INT8 calibration images...")
        print(f"{'='*60}")

        image_paths = []

        # Option 1: Load images from a directory
        if self.args.calibration_dir:
            cal_dir = Path(self.args.calibration_dir)
            if not cal_dir.exists():
                raise FileNotFoundError(f"Calibration directory not found: {cal_dir}")
            # Collect image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                image_paths.extend(cal_dir.glob(ext))
                image_paths.extend(cal_dir.glob(ext.upper()))
            image_paths = sorted(image_paths)
            print(f"  Found {len(image_paths)} images in {cal_dir}")

        # Option 2: Load images from dataset YAML
        elif self.args.data:
            import yaml
            data_yaml_path = check_file(self.args.data)
            yaml_dir = Path(data_yaml_path).parent
            try:
                with open(data_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                # Get validation or training image paths
                val_rel = data.get('val', data.get('train', ''))
                if isinstance(val_rel, str) and val_rel:
                    # Resolve relative path against YAML file's directory
                    val_dir = (yaml_dir / val_rel).resolve()
                    if not val_dir.exists():
                        print(f"  WARNING: Resolved path not found: {val_dir}, trying as-is...")
                        val_dir = Path(val_rel).resolve()
                    if val_dir.is_dir():
                        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                            image_paths.extend(val_dir.glob(ext))
                            image_paths.extend(val_dir.glob(ext.upper()))
                    elif val_dir.is_file():
                        # Text file with image paths
                        with open(val_dir, 'r') as f:
                            image_paths = [Path(line.strip()) for line in f if line.strip()]
                # Also print dataset info
                nc = data.get('nc', '?')
                names = data.get('names', [])
                print(f"  Dataset: {data_yaml_path} (nc={nc}, classes={len(names) if isinstance(names, list) else '?'})")
                image_paths = sorted(image_paths)
                print(f"  Found {len(image_paths)} images from: {val_dir}")
            except Exception as e:
                print(f"  WARNING: Failed to parse dataset YAML: {e}")
                print(f"  Falling back to synthetic calibration data.")

        # Option 3: Use synthetic data as fallback
        if not image_paths:
            print(f"  WARNING: No calibration images found. Using synthetic random images.")
            print(f"  This may produce suboptimal INT8 quantization quality.")
            print(f"  For best results, provide --data or --calibration-dir with real images.")
            # Generate synthetic calibration images
            num_images = min(self.args.num_calibration_images, 20)
            synthetic = np.random.randint(
                0, 255,
                (num_images, *self.imgsz, 3),
                dtype=np.float32
            )
            print(f"  Generated {num_images} synthetic images, shape={synthetic.shape}")
            return synthetic

        # Preprocess images
        num_images = min(self.args.num_calibration_images, len(image_paths))
        image_paths = image_paths[:num_images]
        print(f"  Using {num_images} images for calibration")
        print(f"  Target size: {self.imgsz}")

        images_bhwc = []
        for i, img_path in enumerate(image_paths):
            try:
                img = self._preprocess_image(str(img_path))
                images_bhwc.append(img)
                if (i + 1) % 50 == 0:
                    print(f"    Processed {i + 1}/{num_images} images...")
            except Exception as e:
                print(f"    WARNING: Failed to process {img_path}: {e}")
                continue

        if not images_bhwc:
            raise RuntimeError("No valid calibration images could be loaded. "
                             "Check your --data or --calibration-dir path.")

        # Stack into single array
        images = np.stack(images_bhwc, axis=0).astype(np.float32)
        print(f"  Calibration data shape: {images.shape} (BHWC)")
        print(f"  Data range: [{images.min():.1f}, {images.max():.1f}]")
        return images

    def _preprocess_image(self, img_path):
        """Preprocess a single image for calibration.

        Resizes the image to model input size using LetterBox padding (preserving aspect ratio),
        matching the YOLOv5-Lite inference preprocessing.

        Args:
            img_path (str): Path to the image file.

        Returns:
            numpy.ndarray: Preprocessed image in HWC format, float32, range [0, 255].
        """
        import cv2

        # Read image
        img0 = cv2.imread(img_path)
        if img0 is None:
            raise ValueError(f"Cannot read image: {img_path}")

        # LetterBox resize (maintains aspect ratio, pads to target size)
        img = letterbox(img0, new_shape=self.imgsz, auto=False)[0]

        # Convert BGR to RGB
        img = img[:, :, ::-1]  # BGR → RGB

        return img.astype(np.float32)

    def convert_to_tflite(self):
        """Convert ONNX model to TensorFlow Lite via onnx2tf.

        This mirrors the ultralytics onnx2saved_model() function which:
        1. Installs/verifies dependencies (tensorflow, onnx2tf, etc.)
        2. Prepares INT8 calibration data if needed
        3. Calls onnx2tf.convert() with appropriate parameters
        4. Renames/manages output TFLite files
        5. Adds metadata to TFLite files

        onnx2tf.convert() generates multiple TFLite files:
        - model_float32.tflite: FP32 inference
        - model_dynamic_range_quant.tflite: INT8 weights, FP32 activations
        - model_full_integer_quant.tflite: INT8 weights + INT8 activations + INT8 I/O
        - model_integer_quant_with_int16_act.tflite: INT8 weights + INT16 activations
        """
        print(f"\n{'='*60}")
        print(f"Converting ONNX to TensorFlow Lite...")
        print(f"{'='*60}")

        # Check dependencies
        self._check_tf_dependencies()

        import tensorflow as tf
        print(f"  TensorFlow version: {tf.__version__}")

        # Patch onnx.helper for onnx_graphsurgeon compatibility with ONNX >= 1.17
        import onnx.helper
        if not hasattr(onnx.helper, "float32_to_bfloat16"):
            import struct
            def float32_to_bfloat16(fval):
                ival = struct.unpack("=I", struct.pack("=f", fval))[0]
                return ival >> 16
            onnx.helper.float32_to_bfloat16 = float32_to_bfloat16
            print(f"  Patched onnx.helper.float32_to_bfloat16 for compatibility")

        # Prepare calibration data for INT8 quantization
        np_data = None
        calib_npy_path = None
        if self.args.int8:
            print(f"\n  Preparing INT8 calibration data...")
            calib_images = self.collect_calibration_images()
            calib_npy_path = self.output_dir / "tmp_tflite_int8_calibration_images.npy"
            np.save(str(calib_npy_path), calib_images)  # BHWC float32
            print(f"  Calibration data saved: {calib_npy_path}")
            print(f"  Calibration shape: {calib_images.shape}, dtype: {calib_images.dtype}")

            # onnx2tf expects calibration data in format:
            # [["input_name", npy_path, [[[[min_vals]]]], [[[[max_vals]]]]]]
            np_data = [
                ["images",
                 str(calib_npy_path),
                 [[[[0, 0, 0]]]],      # min values (RGB)
                 [[[[255, 255, 255]]]]  # max values (RGB)
                ]
            ]

        # Remove existing saved_model directory
        if self.saved_model_dir.exists():
            shutil.rmtree(self.saved_model_dir)
        self.saved_model_dir.mkdir(parents=True, exist_ok=True)

        # onnx2tf convert
        import onnx2tf
        print(f"\n  onnx2tf version: {onnx2tf.__version__}")
        print(f"  Starting onnx2tf.convert()...")
        print(f"    input_onnx_file_path: {self.f_onnx}")
        print(f"    output_folder_path: {self.saved_model_dir}")
        print(f"    output_integer_quantized_tflite: {self.args.int8}")
        print(f"    full_integer_quant: {self.args.full_integer}")
        print(f"    disable_group_convolution: False (matching ultralytics — only True for tfjs/edgetpu)")

        t_start = time.time()

        # Determine verbosity level
        verbosity = "debug" if self.args.verbose else "error"

        keras_model = onnx2tf.convert(
            input_onnx_file_path=str(self.f_onnx),
            output_folder_path=str(self.saved_model_dir),
            not_use_onnxsim=True,           # Already simplified above
            verbosity=verbosity,
            output_integer_quantized_tflite=self.args.int8,
            custom_input_op_name_np_data_path=np_data,
            enable_batchmatmul_unfold=not self.args.int8,
            output_signaturedefs=True,
            # IMPORTANT: Must be False for TFLite! Setting True causes onnx2tf to
            # decompose every depthwise convolution (groups=N) into N individual
            # Conv2D ops (Split→Conv2D→Concat per channel), exploding the op count.
            # Ultralytics only uses True for tfjs/edgetpu formats, not for tflite.
            # See: ultralytics/utils/export/tensorflow.py lines 66, 156
            disable_group_convolution=False,
        )

        elapsed = time.time() - t_start
        print(f"\n  onnx2tf conversion completed in {elapsed:.1f}s")

        # Post-process TFLite files (mirroring ultralytics' file management)
        if self.args.int8:
            # Clean up calibration npy file
            if calib_npy_path and calib_npy_path.exists():
                calib_npy_path.unlink()
                print(f"  Removed temporary calibration file: {calib_npy_path}")

            # Rename dynamic_range_quant → int8
            for file in self.saved_model_dir.rglob("*_dynamic_range_quant.tflite"):
                new_name = file.with_name(
                    file.stem.replace("_dynamic_range_quant", "_int8") + file.suffix
                )
                file.rename(new_name)
                print(f"  Renamed: {file.name} → {new_name.name}")

            # Delete int16 activation files (not needed)
            for file in self.saved_model_dir.rglob("*_integer_quant_with_int16_act.tflite"):
                file.unlink()
                print(f"  Removed: {file.name}")

            # Handle full_integer_quant files
            for file in self.saved_model_dir.rglob("*_full_integer_quant.tflite"):
                new_name = file.with_name(
                    file.stem.replace("_full_integer_quant", "_full_integer_quant") + file.suffix
                )
                print(f"  Full integer quant model: {file.name}")

        # List all generated TFLite files
        self._list_tflite_files()

        # Add metadata to TFLite files
        for tflite_file in self.saved_model_dir.rglob("*.tflite"):
            self._add_tflite_metadata(tflite_file)

        return keras_model

    def _check_tf_dependencies(self):
        """Check and report TensorFlow / onnx2tf dependencies."""
        print(f"\n  Checking dependencies...")

        # Check tensorflow
        try:
            import tensorflow as tf
            print(f"    tensorflow: {tf.__version__} ✓")
        except ImportError:
            print(f"    tensorflow: MISSING ✗")
            print(f"    Install: pip install tensorflow>=2.0.0,<=2.19.0")
            raise

        # Check onnx2tf
        try:
            import onnx2tf
            print(f"    onnx2tf: {onnx2tf.__version__} ✓")
        except ImportError:
            print(f"    onnx2tf: MISSING ✗")
            print(f"    Install: pip install onnx2tf>=1.26.3,<1.29.0")
            raise

        # Check onnx
        try:
            import onnx
            print(f"    onnx: {onnx.__version__} ✓")
        except ImportError:
            print(f"    onnx: MISSING ✗")
            raise

        # Check onnxruntime
        try:
            import onnxruntime
            print(f"    onnxruntime: {onnxruntime.__version__} ✓")
        except ImportError:
            print(f"    onnxruntime: MISSING ✗")
            raise

        # Pre-download calibration file for onnx2tf (fixes known issue)
        # onnx2tf requires this file for internal calibration during conversion.
        # With numpy 2.x, the file may need allow_pickle=True to load.
        onnx2tf_file = Path("calibration_image_sample_data_20x128x128x3_float32.npy")
        if onnx2tf_file.exists():
            # Ensure the file is numpy 2.x compatible (no pickled objects)
            try:
                data = np.load(str(onnx2tf_file), allow_pickle=True)
                np.save(str(onnx2tf_file), data.astype(np.float32))
                print(f"    onnx2tf calibration sample: {data.shape} ✓")
            except Exception as e:
                print(f"    WARNING: calibration sample corrupt, recreating...")
                onnx2tf_file.unlink(missing_ok=True)

        if not onnx2tf_file.exists():
            # Generate synthetic calibration sample data
            data = np.random.randint(0, 255, (20, 128, 128, 3)).astype(np.float32)
            np.save(str(onnx2tf_file), data)
            print(f"    onnx2tf calibration sample: created {data.shape} ✓")

    def _list_tflite_files(self):
        """List all TFLite files in the saved_model directory with their sizes."""
        print(f"\n  Generated TFLite models:")
        tflite_files = sorted(self.saved_model_dir.rglob("*.tflite"))
        if not tflite_files:
            print(f"    (none found)")
            return

        for file in tflite_files:
            size_mb = file.stat().st_size / 1e6
            # Determine quantization type from filename
            if "full_integer_quant" in file.name:
                qtype = "INT8 Full Integer (INT8 weights + INT8 activations + INT8 I/O)"
            elif "integer_quant" in file.name:
                qtype = "INT8 Integer (INT8 weights + INT8 activations, FP32 I/O)"
            elif "dynamic_range_quant" in file.name:
                qtype = "INT8 Dynamic Range (INT8 weights, FP32 activations)"
            elif "int8" in file.name:
                qtype = "INT8 Dynamic Range"
            elif "float16" in file.name:
                qtype = "FP16 (FP16 weights, FP32 I/O)"
            elif "float32" in file.name:
                qtype = "FP32 (no quantization)"
            else:
                qtype = "Unknown"
            print(f"    {file.name}: {size_mb:.2f} MB [{qtype}]")

    def _add_tflite_metadata(self, tflite_path):
        """Add metadata JSON to a TFLite file as a zip entry.

        Mirrors ultralytics' _add_tflite_metadata() method.
        Writes metadata.json to the TFLite zip archive.
        """
        import zipfile

        metadata = self.metadata.copy()
        # Add export-specific info
        metadata.update({
            "export_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "quantization": "int8" if self.args.int8 else ("float16" if self.args.half else "float32"),
            "full_integer": self.args.full_integer,
            "tflite_file": Path(tflite_path).name,
        })

        try:
            with zipfile.ZipFile(tflite_path, "a", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        except Exception as e:
            print(f"    WARNING: Failed to add metadata to {tflite_path.name}: {e}")

    def export(self):
        """Run the complete TFLite export pipeline.

        Main entry point following ultralytics' Exporter.__call__() flow.

        Returns:
            Path: Path to the primary exported TFLite file.
        """
        t_start = time.time()

        print(f"\n{'='*60}")
        print(f"YOLOv5-Lite TFLite Export Pipeline")
        print(f"{'='*60}")
        print(f"  Model: {self.weights_path}")
        print(f"  Image size: {self.imgsz}")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  INT8: {self.args.int8}")
        print(f"  Full Integer: {self.args.full_integer}")
        print(f"  FP16: {self.args.half}")
        print(f"  Device: {self.device}")

        # Step 1-2: Load and prepare model
        self.load_model()

        # Note: attempt_load() already calls model.fuse() internally.
        # Calling fuse() again would fail on already-fused Shuffle_Block modules.

        # Step 4: Prepare for ONNX export
        self.prepare_for_export()

        # Step 5: Export ONNX
        f_onnx = self.export_onnx()

        if self.args.onnx_only:
            elapsed = time.time() - t_start
            print(f"\n{'='*60}")
            print(f"ONNX export complete ({elapsed:.1f}s)")
            print(f"ONNX model: {f_onnx}")
            print(f"{'='*60}")
            return f_onnx

        # Step 6-8: Convert to TFLite via onnx2tf
        self.convert_to_tflite()

        # Summarize
        elapsed = time.time() - t_start
        print(f"\n{'='*60}")
        print(f"TFLite export complete ({elapsed:.1f}s)")
        print(f"{'='*60}")
        print(f"\nOutput directory: {self.saved_model_dir}")
        print(f"\nQuantization summary:")
        if self.args.int8:
            if self.args.full_integer:
                print(f"  • Full Integer Quantization (INT8)")
                print(f"    - Weights: INT8")
                print(f"    - Activations: INT8")
                print(f"    - Input: INT8 (quantized)")
                print(f"    - Output: INT8 (quantized)")
                print(f"    - Model: *_full_integer_quant.tflite")
            else:
                print(f"  • Dynamic Range Quantization (INT8)")
                print(f"    - Weights: INT8")
                print(f"    - Activations: FP32 (dynamic quantized at runtime)")
                print(f"    - Input: FP32")
                print(f"    - Output: FP32")
                print(f"    - Model: *_int8.tflite")
        elif self.args.half:
            print(f"  • FP16 Quantization")
            print(f"    - Weights: FP16")
            print(f"    - Activations: FP16")
        else:
            print(f"  • FP32 (no quantization)")
            print(f"    - Model: *_float32.tflite")

        print(f"\nAvailable TFLite models:")
        for tflite_file in sorted(self.saved_model_dir.rglob("*.tflite")):
            size_mb = tflite_file.stat().st_size / 1e6
            print(f"  {tflite_file} ({size_mb:.2f} MB)")

        print(f"\nPython Inference Example:")
        print(f"  import tensorflow as tf")
        print(f"  interpreter = tf.lite.Interpreter(model_path='{self.saved_model_dir}'")
        print(f"  interpreter.allocate_tensors()")
        print(f"  input_details = interpreter.get_input_details()")
        print(f"  output_details = interpreter.get_output_details()")
        print(f"  # Input: {self.imgsz} RGB image, normalized to [0, 255]")

        print(f"\n{'='*60}")
        return str(f_onnx)


def main():
    """Main entry point."""
    args = parse_args()
    set_logging()

    exporter = TFLiteExporter(args)
    exporter.export()


if __name__ == "__main__":
    main()
