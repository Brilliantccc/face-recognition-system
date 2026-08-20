"""
识别测试脚本
测试门禁系统的识别效果
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import torch_utils
from common.database import Database
from common.user_manager import UserManager
from common.config import DATABASE_PATH, FACES_DIR

torch = torch_utils.torch
TORCH_AVAILABLE = torch_utils.TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from trainer.models.facenet import MobileFaceNet


def test_recognition(model_dir="trainer/models/saved/v1"):
    """测试识别效果"""
    print("=" * 60)
    print("识别测试工具")
    print("=" * 60)
    
    # 加载模型
    model_path = os.path.join(model_dir, "inference_model.pth")
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在: {model_path}")
        return
    
    device = torch_utils.get_device(prefer_gpu=True)
    checkpoint = torch.load(model_path, map_location=device)
    
    model = MobileFaceNet(
        embedding_size=checkpoint['embedding_size'],
        num_classes=None
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 模型加载成功: {model_dir}")
    
    # 加载数据库
    db = Database(DATABASE_PATH)
    user_manager = UserManager(db, FACES_DIR)
    
    # 获取所有用户
    users = db.get_all_users(active_only=True)
    print(f"📊 数据库用户数: {len(users)}")
    
    # 测试每个用户
    print("\n" + "=" * 60)
    print("测试结果:")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for user in users:
        user_id = user['id']
        name = user['name']
        
        # 获取用户的人脸图片
        user_dir = os.path.join(FACES_DIR, f"{user_id}_{name}")
        if not os.path.exists(user_dir):
            print(f"  ⚠️ {name}: 图片目录不存在")
            fail_count += 1
            continue
        
        # 读取第一张图片
        images = [f for f in os.listdir(user_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"  ⚠️ {name}: 无图片")
            fail_count += 1
            continue
        
        # 测试识别
        img_path = os.path.join(user_dir, images[0])
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"  ⚠️ {name}: 图片读取失败")
            fail_count += 1
            continue
        
        # 获取嵌入向量
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = torch_utils.Image.fromarray(rgb_img)
        transform = torch_utils.transforms.Compose([
            torch_utils.transforms.Resize((112, 112)),
            torch_utils.transforms.ToTensor(),
            torch_utils.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model(tensor).cpu().numpy().flatten()
        
        # 与数据库中的特征比对
        encodings = db.get_user_face_encodings(user_id)
        if encodings:
            db_embedding = np.frombuffer(encodings[0]['encoding'], dtype=np.float32)
            similarity = np.dot(embedding, db_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(db_embedding)
            )
            
            if similarity > 0.5:
                print(f"  ✅ {name}: 相似度 {similarity:.2%}")
                success_count += 1
            else:
                print(f"  ⚠️ {name}: 相似度 {similarity:.2%} (低于阈值)")
                fail_count += 1
        else:
            print(f"  ⚠️ {name}: 无数据库特征")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print("测试总结:")
    print("=" * 60)
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  成功率: {success_count/(success_count+fail_count)*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="识别测试工具")
    parser.add_argument("--model-dir", type=str, default="trainer/models/saved/v1",
                       help="模型目录")
    
    args = parser.parse_args()
    test_recognition(args.model_dir)
