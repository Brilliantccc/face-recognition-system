"""
YOLO 人脸检测测试脚本
测试 YOLO 集成是否正常工作
"""

import sys
import os
import cv2
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_yolo_detector():
    """测试 YOLO 检测器"""
    print("=" * 50)
    print("YOLO 人脸检测测试")
    print("=" * 50)

    try:
        from common.yolo_detector import YOLOFaceDetector

        print("\n1. 初始化 YOLO 检测器...")
        detector = YOLOFaceDetector(model_size="n", confidence=0.5)
        print("   ✓ YOLO 检测器初始化成功")

        print("\n2. 打开摄像头...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("   ✗ 无法打开摄像头")
            return False

        print("   ✓ 摄像头打开成功")

        print("\n3. 开始检测 (按 ESC 退出)...")
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测人脸
            detect_start = time.time()
            faces = detector.detect(frame)
            detect_time = (time.time() - detect_start) * 1000

            # 绘制检测结果
            for x1, y1, x2, y2, conf in faces:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"Face: {conf:.2%}"
                cv2.putText(frame, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 显示 FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            info_text = f"FPS: {fps:.1f} | Detect: {detect_time:.1f}ms | Faces: {len(faces)}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("YOLO Face Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n   ✓ 测试完成")
        print(f"   平均 FPS: {fps:.1f}")
        return True

    except ImportError as e:
        print(f"\n   ✗ 导入错误: {e}")
        print("   请安装 ultralytics: pip install ultralytics")
        return False
    except Exception as e:
        print(f"\n   ✗ 测试失败: {e}")
        return False


def test_access_control_with_yolo():
    """测试门禁系统 YOLO 集成"""
    print("\n" + "=" * 50)
    print("门禁系统 YOLO 集成测试")
    print("=" * 50)

    try:
        from gate.access_control import AccessControl
        from common.user_manager import UserManager

        print("\n1. 初始化用户管理器...")
        user_manager = UserManager()

        print("\n2. 初始化门禁控制器 (YOLO 模式)...")
        access_control = AccessControl(
            user_manager=user_manager,
            use_yolo=True,
            yolo_model_size="n"
        )

        detection_method = access_control.get_detection_method()
        print(f"   ✓ 检测方法: {detection_method}")

        print("\n3. 启动处理线程...")
        access_control.start_processing()
        time.sleep(2)

        print("\n4. 停止处理...")
        access_control.stop_processing()

        print("\n   ✓ 门禁系统 YOLO 集成测试完成")
        return True

    except Exception as e:
        print(f"\n   ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("人脸识别系统 - YOLO 集成测试")
    print("=" * 50)

    # 测试 YOLO 检测器
    yolo_ok = test_yolo_detector()

    # 测试门禁系统集成
    if yolo_ok:
        access_ok = test_access_control_with_yolo()
    else:
        access_ok = False

    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"YOLO 检测器: {'✓ 通过' if yolo_ok else '✗ 失败'}")
    print(f"门禁系统集成: {'✓ 通过' if access_ok else '✗ 失败'}")

    if yolo_ok and access_ok:
        print("\n✓ 所有测试通过！YOLO 集成成功。")
    else:
        print("\n✗ 部分测试失败，请检查错误信息。")


if __name__ == "__main__":
    main()
