# Real-time-Moving-Small-Object-Detection
###介紹
1. 建置一套可於實際球場環境運行的即時羽球偵測系統  
2.設計結合 attention 與多尺度特徵提取的偵測模型，以應對高速移動與運動模糊情境  
3.採用 heatmap regression 進行位置預測，提升定位穩定度與準確性  
4.透過 ONNX → TensorRT（INT8）將最佳化模型部署於 NVIDIA Jetson Orin Nano  
5.於邊緣裝置上達成約 27 FPS 的即時影像串流推論效能  
