/**
 * @file yolov5_lite_tflite.cpp
 * @brief Implementation of YOLOv5-Lite TFLite full-integer inference.
 *
 * Complete pipeline:
 *   Image (uint8 RGB) → Preprocess (letterbox + INT8 quantize)
 *   → TFLite Interpreter::Invoke()
 *   → Postprocess (INT8 dequant → decode boxes → NMS) → vector<Detection>
 */

#include "yolov5_lite_tflite.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>

// TensorFlow Lite
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif

// ===================================================================
//  Construction / Destruction
// ===================================================================

YOLOv5LiteTFLite::YOLOv5LiteTFLite() = default;
YOLOv5LiteTFLite::~YOLOv5LiteTFLite() = default;

// ===================================================================
//  Model Loading
// ===================================================================

bool YOLOv5LiteTFLite::LoadModel(const std::string& model_path,
                                  DelegateType delegate,
                                  int num_threads) {
    delegate_type_ = delegate;

    // --- 1. Load FlatBuffer model ---
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model_) {
        std::cerr << "[ERROR] Failed to load model: " << model_path << "\n";
        return false;
    }
    std::cout << "[INFO] Model loaded: " << model_path
              << " (" << (model_->GetModel()->subgraphs()->Get(0)->operators()->size())
              << " ops)\n";

    // --- 2. Build resolver with builtin ops ---
    tflite::ops::builtin::BuiltinOpResolver resolver;

    // --- 3. Create interpreter ---
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interpreter_);
    if (!interpreter_) {
        std::cerr << "[ERROR] Failed to build interpreter\n";
        return false;
    }

    // --- 4. Set thread count ---
    interpreter_->SetNumThreads(num_threads);
    std::cout << "[INFO] Threads: " << num_threads << "\n";

    // --- 5. Apply delegate ---
    switch (delegate) {
        case DelegateType::kXNNPACK: {
            TfLiteXNNPACKDelegateOptions xnnpack_opts =
                TfLiteXNNPACKDelegateOptionsDefault();
            xnnpack_opts.num_threads = num_threads;
            auto* xnnpack_delegate = TfLiteXNNPACKDelegateCreate(&xnnpack_opts);
            if (xnnpack_delegate) {
                interpreter_->ModifyGraphWithDelegate(xnnpack_delegate);
                std::cout << "[INFO] XNNPACK delegate applied\n";
            } else {
                std::cerr << "[WARN] XNNPACK delegate unavailable, using CPU\n";
            }
            break;
        }
        case DelegateType::kGPU:
#if defined(__ANDROID__)
        {
            auto gpu_opts = TfLiteGpuDelegateOptionsV2Default();
            gpu_opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
            auto* gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
            if (gpu_delegate) {
                interpreter_->ModifyGraphWithDelegate(gpu_delegate);
                std::cout << "[INFO] GPU delegate applied\n";
            } else {
                std::cerr << "[WARN] GPU delegate unavailable, falling back to XNNPACK\n";
                // fallback to XNNPACK
                TfLiteXNNPACKDelegateOptions xopts = TfLiteXNNPACKDelegateOptionsDefault();
                auto* xd = TfLiteXNNPACKDelegateCreate(&xopts);
                interpreter_->ModifyGraphWithDelegate(xd);
            }
        }
#else
            std::cerr << "[WARN] GPU delegate not available on this platform\n";
            return false;
#endif
            break;

        case DelegateType::kNNAPI:
#if defined(__ANDROID__)
        {
            auto* nnapi_delegate = tflite::NnApiDelegate();
            if (nnapi_delegate) {
                interpreter_->ModifyGraphWithDelegate(nnapi_delegate);
                std::cout << "[INFO] NNAPI delegate applied\n";
            }
        }
#else
            std::cerr << "[WARN] NNAPI not available on this platform\n";
            return false;
#endif
            break;

        case DelegateType::kNone:
        default:
            std::cout << "[INFO] No delegate (plain CPU inference)\n";
            break;
    }

    // --- 6. Allocate tensors ---
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        std::cerr << "[ERROR] Failed to allocate tensors\n";
        return false;
    }

    // --- 7. Verify I/O shapes ---
    auto* input_tensor  = interpreter_->input_tensor(0);
    auto* output_tensor = interpreter_->output_tensor(0);

    if (input_tensor->dims->size != 4) {
        std::cerr << "[ERROR] Expected 4D input, got " << input_tensor->dims->size << "D\n";
        return false;
    }

    input_size_ = input_tensor->dims->data[1];  // height
    std::cout << "[INFO] Input:  [1, " << input_tensor->dims->data[1]
              << ", " << input_tensor->dims->data[2]
              << ", " << input_tensor->dims->data[3] << "] "
              << TfLiteTypeGetName(input_tensor->type)
              << "  scale=" << input_tensor->params.scale
              << "  zp=" << input_tensor->params.zero_point << "\n";

    std::cout << "[INFO] Output: [1, " << output_tensor->dims->data[1]
              << ", " << output_tensor->dims->data[2] << "] "
              << TfLiteTypeGetName(output_tensor->type)
              << "  scale=" << output_tensor->params.scale
              << "  zp=" << output_tensor->params.zero_point << "\n";

    // --- 8. Read quantization params from loaded model ---
    input_scale_       = input_tensor->params.scale;
    input_zero_point_  = input_tensor->params.zero_point;
    output_scale_      = output_tensor->params.scale;
    output_zero_point_ = output_tensor->params.zero_point;
    num_output_dets_   = output_tensor->dims->data[1];
    num_output_fields_ = output_tensor->dims->data[2];
    num_classes_       = num_output_fields_ - 5;  // tx,ty,tw,th,obj + N class scores

    std::cout << "[INFO] Classes: " << num_classes_
              << ", output detections: " << num_output_dets_
              << ", fields: " << num_output_fields_ << "\n";

    // --- 9. Build anchor grids ---
    SetAnchorsFor4Class();
    BuildGrids();

    return true;
}

// ===================================================================
//  Anchor Configuration
// ===================================================================

void YOLOv5LiteTFLite::SetAnchorsFor4Class() {
    // Anchors from v5Lite-s config (pixel coordinates at input_size=320).
    // These are the anchor values BEFORE stride normalization.
    // If your model uses different anchors (e.g., from auto-anchor),
    // update these values accordingly.
    heads_ = {
        // Head 0: stride 8, P3/8
        {8,  {{{10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}}}},
        // Head 1: stride 16, P4/16
        {16, {{{30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}}}},
        // Head 2: stride 32, P5/32
        {32, {{{116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}}}},
    };
}

// ===================================================================
//  Grid Pre-computation
// ===================================================================

void YOLOv5LiteTFLite::BuildGrids() {
    grids_.clear();
    int offset = 0;

    for (const auto& head : heads_) {
        HeadGrid hg;
        hg.stride  = head.stride;
        hg.grid_w  = input_size_ / head.stride;
        hg.grid_h  = input_size_ / head.stride;
        hg.offset  = offset;
        hg.anchors = head.anchors;

        int na = static_cast<int>(head.anchors.size());
        int cells = hg.grid_w * hg.grid_h;
        int total = na * cells;

        // Pre-compute grid positions
        hg.gx.resize(total);
        hg.gy.resize(total);

        for (int a = 0; a < na; ++a) {
            for (int y = 0; y < hg.grid_h; ++y) {
                for (int x = 0; x < hg.grid_w; ++x) {
                    int idx = a * cells + y * hg.grid_w + x;
                    hg.gx[idx] = static_cast<float>(x);
                    hg.gy[idx] = static_cast<float>(y);
                }
            }
        }

        grids_.push_back(hg);
        offset += total;

        std::cout << "[INFO] Head P" << (3 + grids_.size()) << "/"
                  << head.stride
                  << ": grid=" << hg.grid_w << "x" << hg.grid_h
                  << "  anchors=" << na
                  << "  dets=" << total
                  << "  offset=" << hg.offset << "\n";
    }
}

// ===================================================================
//  Preprocessing: Image → INT8 Model Input
// ===================================================================

std::vector<int8_t> YOLOv5LiteTFLite::Preprocess(
        const uint8_t* image_data,
        int width, int height, int stride_bytes,
        float& scale_x, float& scale_y,
        int& pad_left, int& pad_top) {

    // Step 1: Letterbox resize (preserving aspect ratio)
    float r = std::min(static_cast<float>(input_size_) / width,
                       static_cast<float>(input_size_) / height);
    int new_w = static_cast<int>(width * r);
    int new_h = static_cast<int>(height * r);

    // Make even (required by some hardware)
    new_w = (new_w / 2) * 2;
    new_h = (new_h / 2) * 2;

    pad_left = (input_size_ - new_w) / 2;
    pad_top  = (input_size_ - new_h) / 2;

    scale_x = static_cast<float>(width)  / new_w;
    scale_y = static_cast<float>(height) / new_h;

    // Step 2: Resize + pad + INT8 quantize
    std::vector<int8_t> input_tensor(input_size_ * input_size_ * 3, 0);

    // Fill padded area with "grey" (zero after quantization: -zp ≈ 128/255 grey)
    // For zp=-128 and scale=0.003922, grey(128) = round(128*0.003922) - 128 = 0 - 128 = -128
    // Actually grey(128) = 128/255 ≈ 0.502, quantized: round(0.502/0.003922) + (-128) = 128-128 = 0
    // Wait: int8 = round(float/scale) + zp. For scale=0.003922, zp=-128:
    //   pixel=114 (grey for letterbox): float=114/255=0.447, int8=round(0.447/0.003922)-128 = 114-128 = -14
    //   pixel=128: float=0.502, int8=128-128=0
    // We use pixel=114 (standard YOLO letterbox pad value)
    const int8_t pad_pixel_int8 = static_cast<int8_t>(
        std::clamp(static_cast<int>(std::round(114.0f / 255.0f / input_scale_)) + input_zero_point_,
                   -128, 127));

    // Simple nearest-neighbor resize + pad (production code should use bilinear)
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < input_size_; ++y) {
            for (int x = 0; x < input_size_; ++x) {
                int out_idx = (y * input_size_ + x) * 3 + c;
                int8_t val;

                if (x < pad_left || x >= pad_left + new_w ||
                    y < pad_top  || y >= pad_top  + new_h) {
                    val = pad_pixel_int8;
                } else {
                    // Map output (x,y) back to source image
                    int src_x = static_cast<int>((x - pad_left) / r);
                    int src_y = static_cast<int>((y - pad_top)  / r);
                    src_x = std::clamp(src_x, 0, width  - 1);
                    src_y = std::clamp(src_y, 0, height - 1);

                    uint8_t pixel = image_data[src_y * stride_bytes + src_x * 3 + c];
                    // Quantize: float_val = pixel / 255.0
                    //          int8_val = round(float_val / scale) + zero_point
                    float fp = pixel / 255.0f;
                    int32_t quant = static_cast<int32_t>(std::round(fp / input_scale_)) + input_zero_point_;
                    val = static_cast<int8_t>(std::clamp(quant, -128, 127));
                }
                input_tensor[out_idx] = val;
            }
        }
    }

    return input_tensor;
}

// ===================================================================
//  Detection (main inference entry point)
// ===================================================================

std::vector<Detection> YOLOv5LiteTFLite::Detect(
        const uint8_t* image_data, int width, int height, int stride_bytes) {

    // --- Preprocess ---
    float scale_x, scale_y;
    int pad_left, pad_top;
    auto input_buffer = Preprocess(image_data, width, height, stride_bytes,
                                   scale_x, scale_y, pad_left, pad_top);

    // --- Copy input to TFLite tensor ---
    auto* input_tensor = interpreter_->input_tensor(0);
    std::memcpy(interpreter_->typed_input_tensor<int8_t>(0),
                input_buffer.data(),
                input_buffer.size() * sizeof(int8_t));

    // --- Inference ---
    if (interpreter_->Invoke() != kTfLiteOk) {
        std::cerr << "[ERROR] Inference failed\n";
        return {};
    }

    // --- Get output ---
    const int8_t* output_data = interpreter_->typed_output_tensor<int8_t>(0);

    // --- Postprocess ---
    auto detections = Postprocess(output_data);

    // --- Scale boxes back to original image coordinates ---
    for (auto& det : detections) {
        det.x1 = (det.x1 - pad_left) * scale_x;
        det.y1 = (det.y1 - pad_top)  * scale_y;
        det.x2 = (det.x2 - pad_left) * scale_x;
        det.y2 = (det.y2 - pad_top)  * scale_y;

        // Clamp to image bounds
        det.x1 = std::clamp(det.x1, 0.0f, static_cast<float>(width));
        det.y1 = std::clamp(det.y1, 0.0f, static_cast<float>(height));
        det.x2 = std::clamp(det.x2, 0.0f, static_cast<float>(width));
        det.y2 = std::clamp(det.y2, 0.0f, static_cast<float>(height));
    }

    return detections;
}

// ===================================================================
//  Postprocessing: INT8 output → decoded boxes → NMS
// ===================================================================

std::vector<Detection> YOLOv5LiteTFLite::Postprocess(const int8_t* output_data) {
    std::vector<Detection> all_detections;

    // Process each detection head
    for (size_t h = 0; h < grids_.size(); ++h) {
        const auto& hg = grids_[h];
        int na = static_cast<int>(hg.anchors.size());
        int cells = hg.grid_w * hg.grid_h;
        int total = na * cells;

        for (int det_idx = 0; det_idx < total; ++det_idx) {
            int out_idx = (hg.offset + det_idx) * num_output_fields_;

            // --- Dequantize INT8 → float ---
            // float_val = (int8_val - zero_point) * scale
            auto dequant = [this](int8_t v) -> float {
                return (static_cast<float>(v) - output_zero_point_) * output_scale_;
            };

            float tx  = dequant(output_data[out_idx + 0]);
            float ty  = dequant(output_data[out_idx + 1]);
            float tw  = dequant(output_data[out_idx + 2]);
            float th  = dequant(output_data[out_idx + 3]);
            float obj = dequant(output_data[out_idx + 4]);

            // Confidence threshold check
            if (obj < conf_threshold_) continue;

            // Get class scores
            float max_cls = 0.0f;
            int   best_cls = 0;
            for (int c = 0; c < num_classes_; ++c) {
                float score = dequant(output_data[out_idx + 5 + c]);
                if (score > max_cls) {
                    max_cls = score;
                    best_cls = c;
                }
            }

            float confidence = obj * max_cls;
            if (confidence < conf_threshold_) continue;

            // --- Decode box ---
            int anchor_idx = det_idx / cells;
            int cell_idx   = det_idx % cells;

            float gx = hg.gx[det_idx];
            float gy = hg.gy[det_idx];
            float aw = hg.anchors[anchor_idx][0];
            float ah = hg.anchors[anchor_idx][1];

            // YOLOv5 decode (values already sigmoid from model's cat_forward)
            float cx = (tx * 2.0f - 0.5f + gx) * hg.stride;
            float cy = (ty * 2.0f - 0.5f + gy) * hg.stride;
            float bw = std::pow(tw * 2.0f, 2.0f) * aw;
            float bh = std::pow(th * 2.0f, 2.0f) * ah;

            Detection det;
            det.x1         = cx - bw * 0.5f;
            det.y1         = cy - bh * 0.5f;
            det.x2         = cx + bw * 0.5f;
            det.y2         = cy + bh * 0.5f;
            det.confidence = confidence;
            det.class_id   = best_cls;

            all_detections.push_back(det);
        }
    }

    // Apply NMS
    return ApplyNMS(all_detections, nms_threshold_, max_detections_);
}

// ===================================================================
//  Non-Maximum Suppression
// ===================================================================

std::vector<Detection> YOLOv5LiteTFLite::ApplyNMS(
        std::vector<Detection>& detections,
        float iou_threshold,
        int max_detections) {

    // Sort by confidence descending
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<Detection> results;
    std::vector<bool> suppressed(detections.size(), false);

    auto iou = [](const Detection& a, const Detection& b) -> float {
        float inter_x1 = std::max(a.x1, b.x1);
        float inter_y1 = std::max(a.y1, b.y1);
        float inter_x2 = std::min(a.x2, b.x2);
        float inter_y2 = std::min(a.y2, b.y2);

        if (inter_x2 <= inter_x1 || inter_y2 <= inter_y1) return 0.0f;

        float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
        return inter_area / (area_a + area_b - inter_area);
    };

    for (size_t i = 0; i < detections.size() && results.size() < static_cast<size_t>(max_detections); ++i) {
        if (suppressed[i]) continue;

        results.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            if (detections[i].class_id == detections[j].class_id &&
                iou(detections[i], detections[j]) >= iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return results;
}

// ===================================================================
//  Low-level box decoding (used internally by Postprocess)
// ===================================================================

Detection YOLOv5LiteTFLite::DecodeBox(
        int /*head_idx*/, int /*anchor_idx*/,
        int gx, int gy,
        float tx, float ty, float tw, float th,
        float obj_conf, const float* class_scores) {

    // Uses the first head's stride as default; overridden in Postprocess per-head.
    // This function is kept for fine-grained decoding if needed.
    int stride = heads_[0].stride;
    float aw = heads_[0].anchors[0][0];
    float ah = heads_[0].anchors[0][1];

    float cx = (tx * 2.0f - 0.5f + static_cast<float>(gx)) * stride;
    float cy = (ty * 2.0f - 0.5f + static_cast<float>(gy)) * stride;
    float bw = std::pow(tw * 2.0f, 2.0f) * aw;
    float bh = std::pow(th * 2.0f, 2.0f) * ah;

    float max_cls = 0.0f;
    int   best_cls = 0;
    for (int c = 0; c < num_classes_; ++c) {
        if (class_scores[c] > max_cls) {
            max_cls = class_scores[c];
            best_cls = c;
        }
    }

    Detection det;
    det.x1         = cx - bw * 0.5f;
    det.y1         = cy - bh * 0.5f;
    det.x2         = cx + bw * 0.5f;
    det.y2         = cy + bh * 0.5f;
    det.confidence = obj_conf * max_cls;
    det.class_id   = best_cls;
    return det;
}
