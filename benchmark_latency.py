import time, json, os
import numpy as np
import torch
from ultralytics import YOLO

WEIGHTS = 'runs/detect/runs/full_finetuned/weights/best.pt'
IMGSZ = 640
WARMUP = 20
RUNS = 200
OUT = 'analysis/latency_benchmark.json'

def bench(model, dummy, label):
    for _ in range(WARMUP):
        model.predict(dummy, imgsz=IMGSZ, verbose=False, device=0)
    torch.cuda.synchronize()
    times = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=IMGSZ, verbose=False, device=0)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    stats = {'engine': label,
             'mean_ms': round(float(times.mean()), 2),
             'p50_ms': round(float(np.percentile(times, 50)), 2),
             'p99_ms': round(float(np.percentile(times, 99)), 2),
             'fps': round(1000.0 / float(times.mean()), 1)}
    print(f"{label:<18} mean {stats['mean_ms']:>6.2f} ms | p99 {stats['p99_ms']:>6.2f} ms | {stats['fps']:>6.1f} FPS")
    return stats

def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Benchmarking at {IMGSZ}x{IMGSZ}, {RUNS} runs, {WARMUP} warm-up\n")
    dummy = (np.random.rand(IMGSZ, IMGSZ, 3) * 255).astype(np.uint8)
    results = []
    print("Loading PyTorch model...")
    pt = YOLO(WEIGHTS)
    results.append(bench(pt, dummy, 'PyTorch FP32'))
    try:
        print("\nExporting TensorRT FP16 engine (takes a few minutes)...")
        engine_path = YOLO(WEIGHTS).export(format='engine', half=True, imgsz=IMGSZ, device=0)
        trt = YOLO(engine_path)
        results.append(bench(trt, dummy, 'TensorRT FP16'))
    except Exception as e:
        print(f"  TensorRT export failed ({type(e).__name__}: {e}). Falling back to ONNX...")
        try:
            onnx_path = YOLO(WEIGHTS).export(format='onnx', imgsz=IMGSZ, device=0)
            ox = YOLO(onnx_path)
            results.append(bench(ox, dummy, 'ONNX'))
        except Exception as e2:
            print(f"  ONNX export also failed ({type(e2).__name__}: {e2}).")
    print("\n=== Summary ===")
    base = next((r for r in results if r['engine'] == 'PyTorch FP32'), None)
    for r in results:
        line = f"{r['engine']:<18} {r['fps']:>6.1f} FPS  (mean {r['mean_ms']} ms, p99 {r['p99_ms']} ms)"
        if base and r is not base:
            line += f"  -> {r['fps']/base['fps']:.2f}x faster"
        print(line)
    os.makedirs('analysis', exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump({'gpu': torch.cuda.get_device_name(0), 'imgsz': IMGSZ, 'runs': RUNS, 'results': results}, f, indent=2)
    print(f"\nSaved {OUT}")

if __name__ == '__main__':
    main()
