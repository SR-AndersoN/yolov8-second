from ultralytics import YOLO

# 加载你的模型
model = YOLO("yolov8n.pt") 

# 训练时直接传入数据增强参数
results = model.train(
    data="/kaggle/working/yolov8-second/MInecraft Player.v2i.yolov8/data.yaml",
    epochs=100,
    imgsz=640,
    # 下面是数据增强参数 (数值代表概率或变化幅度)
    mosaic=1.0,      # 100% 开启马赛克拼接 (默认通常就是开启的)
    hsv_h=0.015,     # 色调变化幅度
    hsv_s=0.7,       # 饱和度变化幅度 (调高点适应MC不同光照)
    hsv_v=0.4,       # 明度变化幅度 (适应夜晚和矿洞)
    degrees=0.0,     # 旋转角度 (如果截图像素比较方正，建议设为0)
    translate=0.1,   # 平移
    scale=0.5,       # 缩放
    fliplr=0.5,      # 50%概率左右翻转
    flipud=0.0       # 0%概率上下翻转 (前面提到的，生物不倒立)
)