# Real-time-Moving-Small-Object-Detection
###介紹
1. 建置一套可於實際球場環境運行的即時羽球偵測系統  
2.設計結合 attention 與多尺度特徵提取的偵測模型，以應對高速移動與運動模糊情境  
3.採用 heatmap regression 進行位置預測，提升定位穩定度與準確性  
4.透過 ONNX → TensorRT（INT8）將最佳化模型部署於 NVIDIA Jetson Orin Nano  
5.於邊緣裝置上達成約 27 FPS 的即時影像串流推論效能  

系統流程:  
<img width="4039" height="662" alt="流程圖" src="https://github.com/user-attachments/assets/788181fe-065c-48cb-b9f0-0c5de590482f" />  

模型規模和運行效能:  
<img width="331" height="269" alt="image" src="https://github.com/user-attachments/assets/da53d7f8-e31e-4ece-afb4-f71f08342606" />  

Demo:  
<img width="846" height="480" alt="羽球偵測圖片" src="https://github.com/user-attachments/assets/16fa9c05-7b57-4f12-9916-a067b39602dc" />  

https://github.com/user-attachments/assets/e0a021f6-d06c-43a3-a949-ebf764685671




