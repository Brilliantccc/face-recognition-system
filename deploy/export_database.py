#!/usr/bin/env python3
"""
导出人脸数据库为 C++ 可读的二进制格式
"""
import sqlite3
import numpy as np
import struct
import os

def export_database(db_path: str, output_path: str):
    """导出数据库为 C++ 可读的二进制格式"""
    print(f"连接数据库: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查询活跃用户的人脸编码
    cursor.execute("""
        SELECT u.name, e.encoding
        FROM users u
        JOIN face_encodings e ON u.id = e.user_id
        WHERE u.is_active = 1
    """)

    names = []
    embeddings = []

    for name, encoding_blob in cursor.fetchall():
        embedding = np.frombuffer(encoding_blob, dtype=np.float32)
        if len(embedding) == 128:
            names.append(name)
            embeddings.append(embedding)
        else:
            print(f"跳过无效编码: {name} (维度: {len(embedding)})")

    conn.close()

    if not names:
        print("没有找到有效的人脸数据")
        return

    print(f"找到 {len(names)} 个有效人脸数据")

    # 写入二进制文件
    embeddings_array = np.array(embeddings, dtype=np.float32)

    with open(output_path, 'wb') as f:
        # 文件头: 版本号, 人数, 维度
        f.write(struct.pack('<III', 1, len(names), 128))

        # 写入所有 embedding
        f.write(embeddings_array.tobytes())

        # 写入所有人名
        for name in names:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

    print(f"数据库导出成功: {output_path}")
    print(f"   - 人数: {len(names)}")
    print(f"   - 嵌入维度: 128")
    print(f"   - 文件大小: {os.path.getsize(output_path)} bytes")

    # 打印导出的人名
    print(f"\n导出的人名:")
    for i, name in enumerate(names, 1):
        print(f"   {i}. {name}")

if __name__ == '__main__':
    db_path = 'data/db/face_access.db'
    output_path = 'deploy/models/embeddings.bin'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    export_database(db_path, output_path)
