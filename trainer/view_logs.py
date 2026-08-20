"""
日志查看工具
查看门禁系统的进出记录
"""

import os
import sys
import sqlite3
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import DATABASE_PATH


def view_logs(db_path=None, days=7, user_name=None):
    """
    查看门禁日志
    :param db_path: 数据库路径
    :param days: 查看最近几天
    :param user_name: 只查看特定用户
    """
    db_path = db_path or DATABASE_PATH
    
    if not os.path.exists(db_path):
        print(f"❌ 数据库不存在: {db_path}")
        return
    
    print("=" * 80)
    print(f"门禁日志 (最近 {days} 天)")
    print("=" * 80)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 查询日志
    query = """
        SELECT user_name, timestamp, result, confidence, gate_name
        FROM access_log
        WHERE timestamp >= datetime('now', ?)
    """
    params = [f'-{days} days']
    
    if user_name:
        query += " AND user_name = ?"
        params.append(user_name)
    
    query += " ORDER BY timestamp DESC"
    
    cursor.execute(query, params)
    logs = cursor.fetchall()
    
    if not logs:
        print("  暂无记录")
        return
    
    # 统计
    total = len(logs)
    success = sum(1 for log in logs if log[2] == 'success')
    fail = total - success
    
    # 显示日志
    print(f"\n{'用户':<15} {'时间':<25} {'结果':<10} {'置信度':<10} {'门禁':<10}")
    print("-" * 80)
    
    for log in logs[:50]:  # 最多显示50条
        name, timestamp, result, confidence, gate = log
        result_icon = "✅" if result == 'success' else "❌"
        conf_str = f"{confidence:.2%}" if confidence else "N/A"
        time_str = timestamp[:19] if timestamp else "N/A"
        
        print(f"{name:<15} {time_str:<25} {result_icon} {result:<8} {conf_str:<10} {gate or 'N/A':<10}")
    
    if total > 50:
        print(f"\n  ... 还有 {total - 50} 条记录")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("统计信息:")
    print("=" * 80)
    print(f"  总记录: {total}")
    print(f"  成功: {success} ({success/total*100:.1f}%)")
    print(f"  失败: {fail} ({fail/total*100:.1f}%)")
    
    # 按用户统计
    print("\n按用户统计:")
    cursor.execute("""
        SELECT user_name, COUNT(*) as cnt
        FROM access_log
        WHERE timestamp >= datetime('now', ?)
        GROUP BY user_name
        ORDER BY cnt DESC
        LIMIT 10
    """, [f'-{days} days'])
    
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} 次")
    
    conn.close()


def export_logs(db_path=None, output_file="access_logs.csv", days=30):
    """导出日志到CSV"""
    db_path = db_path or DATABASE_PATH
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT user_name, timestamp, result, confidence, gate_name
        FROM access_log
        WHERE timestamp >= datetime('now', ?)
        ORDER BY timestamp DESC
    """, [f'-{days} days'])
    
    logs = cursor.fetchall()
    conn.close()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("用户,时间,结果,置信度,门禁\n")
        for log in logs:
            name, timestamp, result, confidence, gate = log
            f.write(f"{name},{timestamp},{result},{confidence or ''},{gate or ''}\n")
    
    print(f"✅ 日志已导出: {output_file}")
    print(f"   共 {len(logs)} 条记录")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="门禁日志查看工具")
    parser.add_argument("--db", type=str, default=None, help="数据库路径")
    parser.add_argument("--days", type=int, default=7, help="查看最近几天")
    parser.add_argument("--user", type=str, default=None, help="只查看特定用户")
    parser.add_argument("--export", type=str, default=None, help="导出到CSV文件")
    
    args = parser.parse_args()
    
    if args.export:
        export_logs(args.db, args.export, args.days)
    else:
        view_logs(args.db, args.days, args.user)
