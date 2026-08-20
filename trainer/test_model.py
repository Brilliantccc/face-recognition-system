"""
测试已训练模型的效果
使用 embedding 相似度进行人脸识别
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import torch_utils

torch = torch_utils.torch
F = torch.nn.functional
transforms = torch_utils.transforms
Image = torch_utils.Image
TORCH_AVAILABLE = torch_utils.TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from trainer.models.facenet import MobileFaceNet


def load_model(model_dir="trainer/models/saved"):
    """加载模型"""
    model_path = os.path.join(model_dir, "inference_model.pth")
    info_path = os.path.join(model_dir, "model_info.json")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None, None, None
    
    # 加载模型
    device = torch_utils.get_device(prefer_gpu=True)
    checkpoint = torch.load(model_path, map_location=device)
    
    model = MobileFaceNet(
        embedding_size=checkpoint['embedding_size'],
        num_classes=None
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载类别信息
    class_to_idx = {}
    idx_to_class = {}
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
            class_to_idx = info.get('class_to_idx', {})
            idx_to_class = {int(k): v for k, v in info.get('idx_to_class', {}).items()}
    
    print(f"✅ 模型加载成功")
    print(f"   设备: {device}")
    print(f"   类别数: {len(idx_to_class)}")
    print(f"   嵌入维度: {checkpoint['embedding_size']}")
    
    return model, idx_to_class, device


def get_embedding(model, image_path, device):
    """获取图片的嵌入向量"""
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 获取嵌入
    with torch.no_grad():
        embedding = model(image_tensor)
    
    return embedding.cpu().numpy().flatten()


def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def test_model_on_dataset(model, device, data_dir="trainer/data/aligned", num_samples=100):
    """在数据集上测试模型"""
    print(f"\n📊 在数据集上测试模型...")
    
    val_dir = os.path.join(data_dir, "val")
    if not os.path.exists(val_dir):
        print(f"❌ 验证集目录不存在: {val_dir}")
        return
    
    # 收集验证集图片
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    samples = []
    
    for person in os.listdir(val_dir):
        person_dir = os.path.join(val_dir, person)
        if not os.path.isdir(person_dir):
            continue
        
        for f in os.listdir(person_dir):
            if os.path.splitext(f)[1].lower() in valid_exts:
                samples.append((os.path.join(person_dir, f), person))
    
    # 随机采样
    if len(samples) > num_samples:
        import random
        samples = random.sample(samples, num_samples)
    
    print(f"   测试样本数: {len(samples)}")
    
    # 获取每类的平均嵌入
    person_embeddings = {}
    for img_path, person in samples:
        embedding = get_embedding(model, img_path, device)
        if person not in person_embeddings:
            person_embeddings[person] = []
        person_embeddings[person].append(embedding)
    
    # 计算类内相似度和类间相似度
    intra_similarities = []
    inter_similarities = []
    
    persons = list(person_embeddings.keys())
    
    for person in persons:
        embs = person_embeddings[person]
        if len(embs) >= 2:
            # 类内相似度
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    sim = cosine_similarity(embs[i], embs[j])
                    intra_similarities.append(sim)
    
    # 计算类间相似度（随机采样）
    import random
    for _ in range(min(1000, len(persons) * 5)):
        p1, p2 = random.sample(persons, 2)
        emb1 = random.choice(person_embeddings[p1])
        emb2 = random.choice(person_embeddings[p2])
        sim = cosine_similarity(emb1, emb2)
        inter_similarities.append(sim)
    
    print(f"\n📈 相似度统计:")
    if intra_similarities:
        print(f"   类内相似度 (同一个人):")
        print(f"     均值: {np.mean(intra_similarities):.4f}")
        print(f"     标准差: {np.std(intra_similarities):.4f}")
        print(f"     最小值: {np.min(intra_similarities):.4f}")
        print(f"     最大值: {np.max(intra_similarities):.4f}")
    
    if inter_similarities:
        print(f"   类间相似度 (不同人):")
        print(f"     均值: {np.mean(inter_similarities):.4f}")
        print(f"     标准差: {np.std(inter_similarities):.4f}")
        print(f"     最小值: {np.min(inter_similarities):.4f}")
        print(f"     最大值: {np.max(inter_similarities):.4f}")
    
    # 计算可分离性
    if intra_similarities and inter_similarities:
        separation = np.mean(intra_similarities) - np.mean(inter_similarities)
        print(f"\n📊 模型评估:")
        print(f"   类内-类间差异: {separation:.4f}")
        
        if separation > 0.3:
            print(f"   ✅ 模型效果良好，类内紧凑，类间分离")
        elif separation > 0.1:
            print(f"   ⚠️ 模型效果一般，建议继续训练")
        else:
            print(f"   ❌ 模型效果较差，需要重新训练")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试已训练模型")
    parser.add_argument("--model-dir", type=str, default="trainer/models/saved",
                       help="模型目录")
    parser.add_argument("--data", type=str, default="trainer/data/aligned",
                       help="数据集目录")
    parser.add_argument("--num-samples", type=int, default=200,
                       help="测试样本数")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("模型测试工具")
    print("=" * 50)
    
    # 加载模型
    model, idx_to_class, device = load_model(args.model_dir)
    if model is None:
        return
    
    # 在数据集上测试
    test_model_on_dataset(model, device, args.data, args.num_samples)


if __name__ == "__main__":
    main()
