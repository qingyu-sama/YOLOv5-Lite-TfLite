/**
 * @file main.cpp
 * @brief Example: YOLOv5-Lite TFLite full-integer inference in C++.
 *
 * Usage:
 *   ./yolov5_lite_tflite \
 *       --model best_full_integer_quant.tflite \
 *       --image test.jpg \
 *       --conf 0.25 --nms 0.45 \
 *       [--delegate xnnpack|gpu|nnapi] \
 *       [--threads 4] \
 *       [--benchmark]
 *
 * Build (Linux):
 *   mkdir build && cd build
 *   cmake .. -DTensorFlow_DIR=/path/to/libtensorflow-lite.so
 *   make
 *
 * Dependencies:
 *   - TensorFlow Lite (libtensorflow-lite.so)
 *   - OpenCV (libopencv_core, libopencv_imgproc, libopencv_imgcodecs)
 *   - C++17 compiler (g++ >= 8 or clang >= 7)
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

// OpenCV for image I/O
#include <opencv2/opencv.hpp>

#include "yolov5_lite_tflite.h"

// ===================================================================
//  Command-line argument parsing (lightweight, no external lib needed)
// ===================================================================
struct Args {
    std::string model_path;
    std::string image_path;
    std::string delegate = "xnnpack";
    float conf_threshold  = 0.25f;
    float nms_threshold   = 0.45f;
    int   num_threads     = 4;
    bool  benchmark       = false;
    int   benchmark_runs  = 100;
};

Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            args.image_path = argv[++i];
        } else if (arg == "--delegate" && i + 1 < argc) {
            args.delegate = argv[++i];
        } else if (arg == "--conf" && i + 1 < argc) {
            args.conf_threshold = std::stof(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            args.nms_threshold = std::stof(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            args.num_threads = std::stoi(argv[++i]);
        } else if (arg == "--benchmark") {
            args.benchmark = true;
        } else if (arg == "--runs" && i + 1 < argc) {
            args.benchmark_runs = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --model <path>     TFLite model file (required)\n"
                      << "  --image <path>     Input image (required)\n"
                      << "  --delegate <type>  xnnpack|gpu|nnapi|none (default: xnnpack)\n"
                      << "  --conf <float>     Confidence threshold (default: 0.25)\n"
                      << "  --nms <float>      NMS IoU threshold (default: 0.45)\n"
                      << "  --threads <int>    CPU threads (default: 4)\n"
                      << "  --benchmark        Run benchmark mode\n"
                      << "  --runs <int>       Benchmark iterations (default: 100)\n"
                      << "  --help, -h         Show this help\n";
            exit(0);
        }
    }
    return args;
}

DelegateType ParseDelegate(const std::string& name) {
    if (name == "xnnpack") return DelegateType::kXNNPACK;
    if (name == "gpu")     return DelegateType::kGPU;
    if (name == "nnapi")   return DelegateType::kNNAPI;
    if (name == "none")    return DelegateType::kNone;
    std::cerr << "[WARN] Unknown delegate '" << name << "', using XNNPACK\n";
    return DelegateType::kXNNPACK;
}

// ===================================================================
//  Visualization
// ===================================================================
void DrawDetections(cv::Mat& image, const std::vector<Detection>& detections,
                    const std::vector<std::string>& class_names) {
    // Distinct colors for 4 classes
    static const cv::Scalar kColors[] = {
        cv::Scalar(0, 0, 255),     // Red    - Drowning
        cv::Scalar(0, 255, 0),     // Green  - Swimming
        cv::Scalar(255, 0, 0),     // Blue   - fishing
        cv::Scalar(255, 255, 0),   // Cyan   - no fishing
    };

    for (const auto& det : detections) {
        cv::Scalar color = kColors[det.class_id % 4];
        cv::Rect box(static_cast<int>(det.x1), static_cast<int>(det.y1),
                     static_cast<int>(det.x2 - det.x1),
                     static_cast<int>(det.y2 - det.y1));
        cv::rectangle(image, box, color, 2);

        std::string label = class_names[det.class_id] + " " +
                            std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(image,
                      cv::Point(det.x1, det.y1 - text_size.height - 4),
                      cv::Point(det.x1 + text_size.width, det.y1),
                      color, cv::FILLED);
        cv::putText(image, label,
                    cv::Point(det.x1, det.y1 - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

// ===================================================================
//  Main
// ===================================================================
int main(int argc, char** argv) {
    auto args = ParseArgs(argc, argv);

    if (args.model_path.empty() || args.image_path.empty()) {
        std::cerr << "ERROR: --model and --image are required. Use --help for usage.\n";
        return 1;
    }

    // ---- 1. Load image ----
    cv::Mat image_bgr = cv::imread(args.image_path);
    if (image_bgr.empty()) {
        std::cerr << "ERROR: Cannot read image: " << args.image_path << "\n";
        return 1;
    }
    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);

    std::cout << "[INFO] Image: " << image_rgb.cols << "x" << image_rgb.rows
              << ", " << image_rgb.channels() << " channels\n";

    // ---- 2. Load model ----
    YOLOv5LiteTFLite detector;
    detector.SetConfidenceThreshold(args.conf_threshold);
    detector.SetNMSThreshold(args.nms_threshold);

    DelegateType delegate = ParseDelegate(args.delegate);
    if (!detector.LoadModel(args.model_path, delegate, args.num_threads)) {
        std::cerr << "ERROR: Failed to load model\n";
        return 1;
    }

    std::cout << "\n[INFO] Model I/O parameters:\n"
              << "  Input:  [1, " << detector.InputHeight()
              << ", " << detector.InputWidth() << ", 3] int8\n"
              << "  Input scale/zp:  " << detector.InputScale()
              << " / " << detector.InputZeroPoint() << "\n"
              << "  Output scale/zp: " << detector.OutputScale()
              << " / " << detector.OutputZeroPoint() << "\n"
              << "  Classes: " << detector.NumClasses() << "\n";

    // ---- 3. Benchmark mode ----
    if (args.benchmark) {
        std::cout << "\n[INFO] Benchmark: " << args.benchmark_runs << " iterations...\n";

        // Warmup
        for (int i = 0; i < 10; ++i) {
            detector.Detect(image_rgb.data, image_rgb.cols, image_rgb.rows,
                            static_cast<int>(image_rgb.step1()));
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < args.benchmark_runs; ++i) {
            detector.Detect(image_rgb.data, image_rgb.cols, image_rgb.rows,
                            static_cast<int>(image_rgb.step1()));
        }
        auto t_end = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double avg_ms = total_ms / args.benchmark_runs;

        std::cout << "\n  Total:   " << total_ms << " ms\n"
                  << "  Average: " << avg_ms << " ms/inference\n"
                  << "  FPS:     " << (1000.0 / avg_ms) << "\n";
    }

    // ---- 4. Single inference ----
    auto t0 = std::chrono::high_resolution_clock::now();
    auto detections = detector.Detect(image_rgb.data, image_rgb.cols, image_rgb.rows,
                                      static_cast<int>(image_rgb.step1()));
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "\n[INFO] Inference: " << ms << " ms\n";
    std::cout << "[INFO] Detections: " << detections.size() << "\n";

    // ---- 5. Display results ----
    const auto& class_names = detector.ClassNames();
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& d = detections[i];
        std::cout << "  [" << i << "] "
                  << class_names[d.class_id]
                  << "  conf=" << d.confidence
                  << "  box=(" << static_cast<int>(d.x1) << "," << static_cast<int>(d.y1)
                  << "," << static_cast<int>(d.x2) << "," << static_cast<int>(d.y2) << ")\n";
    }

    // ---- 6. Save output image ----
    DrawDetections(image_bgr, detections, class_names);
    std::string output_path = "output_detected.jpg";
    cv::imwrite(output_path, image_bgr);
    std::cout << "\n[INFO] Output saved: " << output_path << "\n";

    return 0;
}
