/**
 * @file yolov5_lite_tflite.h
 * @brief C++ TensorFlow Lite inference for YOLOv5-Lite INT8 full-integer model.
 *
 * Model specs (best_full_integer_quant.tflite):
 *   Input:  [1, 320, 320, 3]  int8   scale=0.003922  zero_point=-128
 *   Output: [1, 6300, 9]     int8   scale=0.003921  zero_point=-128
 *   Classes: 4  (Drowning, Swimming, fishing, no fishing)
 *
 * Output format (cat_forward → raw sigmoid values per grid cell):
 *   [tx, ty, tw, th, obj_conf, cls0, cls1, cls2, cls3]
 *
 * Detection heads (input_size=320×320):
 *   Head 0: stride= 8, grid=40×40, anchors=[10,13],[16,30],[33,23]  → 4800 dets
 *   Head 1: stride=16, grid=20×20, anchors=[30,61],[62,45],[59,119] → 1200 dets
 *   Head 2: stride=32, grid=10×10, anchors=[116,90],[156,198],[373,326] → 300 dets
 *   Total: 6300
 *
 * Decoding from grid-relative to pixel coordinates:
 *   cx = (sigmoid(tx) * 2.0 - 0.5 + grid_x) * stride
 *   cy = (sigmoid(ty) * 2.0 - 0.5 + grid_y) * stride
 *   w  = (sigmoid(tw) * 2.0)^2 * anchor_w
 *   h  = (sigmoid(th) * 2.0)^2 * anchor_h
 *
 * NOTE: The model output is ALREADY sigmoid-activated (cat_forward uses x.sigmoid()),
 * so tx/ty/tw/th are in [0, 1] and we do NOT re-apply sigmoid in the decoder.
 *
 * Dependencies:
 *   - TensorFlow Lite C API (tensorflow/lite)
 *   - OpenCV (for image loading/preprocessing)
 *   - C++17
 */

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Forward declarations for TFLite types (avoid heavy includes in header)
// ---------------------------------------------------------------------------
namespace tflite {
class FlatBufferModel;
class Interpreter;
}  // namespace tflite

// ---------------------------------------------------------------------------
// Detection result
// ---------------------------------------------------------------------------
struct Detection {
    float x1;          // top-left x (pixels)
    float y1;          // top-left y (pixels)
    float x2;          // bottom-right x (pixels)
    float y2;          // bottom-right y (pixels)
    float confidence;  // detection confidence [0, 1]
    int   class_id;    // class index
};

// ---------------------------------------------------------------------------
// TFLite delegate type
// ---------------------------------------------------------------------------
enum class DelegateType {
    kNone,      // Plain TFLite CPU
    kXNNPACK,   // XNNPACK accelerated CPU (recommended for most cases)
    kGPU,       // GPU delegate (Android/iOS)
    kNNAPI,     // Android NNAPI delegate
};

// ---------------------------------------------------------------------------
// Detection head configuration
// ---------------------------------------------------------------------------
struct DetectionHead {
    int stride;                               // downsample factor
    std::vector<std::array<float, 2>> anchors;  // anchor (w, h) in pixels
};

// ---------------------------------------------------------------------------
// YOLOv5-Lite TFLite inference engine
// ---------------------------------------------------------------------------
class YOLOv5LiteTFLite {
public:
    YOLOv5LiteTFLite();
    ~YOLOv5LiteTFLite();

    // ---- Model loading ----
    /// Load the TFLite model from file and configure delegates.
    /// @param model_path  Path to .tflite file
    /// @param delegate    Delegate type (default: XNNPACK for CPU acceleration)
    /// @param num_threads Number of CPU threads (default: 4)
    /// @return true on success
    bool LoadModel(const std::string& model_path,
                   DelegateType delegate = DelegateType::kXNNPACK,
                   int num_threads = 4);

    // ---- Inference ----
    /// Run detection on an image.
    /// @param image_data  RGB image data in HWC format, uint8 [0, 255]
    /// @param width       Image width in pixels
    /// @param height      Image height in pixels
    /// @param stride      Row stride in bytes (width * channels for tightly packed)
    /// @return            Detected objects after NMS
    std::vector<Detection> Detect(const uint8_t* image_data,
                                  int width, int height, int stride);

    // ---- Settings ----
    void SetConfidenceThreshold(float conf) { conf_threshold_ = conf; }
    void SetNMSThreshold(float iou)         { nms_threshold_ = iou; }
    void SetMaxDetections(int max_det)      { max_detections_ = max_det; }

    // ---- Metadata access ----
    int    InputWidth()       const { return input_size_; }
    int    InputHeight()      const { return input_size_; }
    int    NumClasses()       const { return num_classes_; }
    float  InputScale()       const { return input_scale_; }
    int32_t InputZeroPoint()  const { return input_zero_point_; }
    float  OutputScale()      const { return output_scale_; }
    int32_t OutputZeroPoint() const { return output_zero_point_; }

    const std::vector<std::string>& ClassNames() const { return class_names_; }

private:
    // ---- Anchor / grid setup ----
    void BuildGrids();
    void SetAnchorsFor4Class();   // Custom anchors for the drowning-detection model

    // ---- Preprocessing ----
    /// Resize image to model input size with letterbox (preserving aspect ratio)
    /// and quantize to INT8.
    std::vector<int8_t> Preprocess(const uint8_t* image_data,
                                   int width, int height, int stride_bytes,
                                   float& scale_x, float& scale_y,
                                   int& pad_left, int& pad_top);

    // ---- Postprocessing ----
    /// Decode INT8 output tensor to floating-point detections + NMS.
    std::vector<Detection> Postprocess(const int8_t* output_data);

    /// Apply NMS to decoded detections.
    static std::vector<Detection> ApplyNMS(std::vector<Detection>& detections,
                                           float iou_threshold,
                                           int max_detections);

    /// Convert raw grid-relative output to pixel-coordinate boxes.
    Detection DecodeBox(int head_idx, int anchor_idx,
                        int gx, int gy,
                        float tx, float ty, float tw, float th,
                        float obj_conf, const float* class_scores);

    // ---- TFLite state ----
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter>     interpreter_;
    DelegateType delegate_type_ = DelegateType::kXNNPACK;

    // ---- Model I/O parameters ----
    int     input_size_        = 320;
    float   input_scale_       = 0.003921568859368563f;
    int32_t input_zero_point_  = -128;
    float   output_scale_      = 0.0039214747957885265f;
    int32_t output_zero_point_ = -128;
    int     num_output_dets_   = 6300;
    int     num_classes_       = 4;
    int     num_output_fields_ = 9;   // tx,ty,tw,th,obj + 4 class scores

    // ---- Detection heads ----
    std::vector<DetectionHead> heads_;
    // Pre-computed grid coordinates for each detection head
    // grid_x_[h][y * grid_w + x] = x
    struct HeadGrid {
        int    stride;
        int    grid_w, grid_h;
        int    offset;  // starting index in the [6300] output
        std::vector<float> gx;  // gx[y * grid_w + x]
        std::vector<float> gy;  // gy[y * grid_w + x]
        std::vector<std::array<float, 2>> anchors;  // anchor (w,h) in pixels
    };
    std::vector<HeadGrid> grids_;

    // ---- Postprocessing parameters ----
    float conf_threshold_  = 0.25f;
    float nms_threshold_   = 0.45f;
    int   max_detections_  = 100;

    // ---- Class names ----
    std::vector<std::string> class_names_ = {
        "Drowning", "Swimming", "fishing", "no fishing"
    };

    // ---- Profiling ----
    bool profile_enabled_ = true;
};
