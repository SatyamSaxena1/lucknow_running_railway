# Install the tensorRT as shown in this video part: https://youtu.be/WoKZDgCvaqI?si=92LL5dA2YFrHfxm7&t=895
# After installation of tensorRT then also install below library
#!pip install onnx onnxsim onnxruntime-gpu

from ultralytics import YOLO
model_path = r"C:\railway_running\lucknow_running\models\arm_medium_90.pt"
# model_path = r"C:\railway_running\lucknow_running\models\best_81_l.pt"

print("Model using : ",model_path)
model = YOLO(model_path)
model.export(format='engine', device=0, half=True, workspace=12,nms=True)
