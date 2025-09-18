#!/usr/bin/env python3
"""
複雜輸出系統 - 任務執行模組

這個模組提供了一個功能豐富的複雜輸出系統，包含：
- 多種數據格式的展示（表格、圖表、JSON、XML）
- 動態統計分析功能
- 彩色終端輸出和格式化
- 模擬實時數據更新
- 多維度數據可視化
- 系統性能監控展示

主要功能：
- 複雜數據表格生成和顯示
- ASCII 圖表繪製
- 實時數據流模擬
- 多種輸出格式支援
- 動態進度條和狀態顯示
- 系統資源監控

技術特點：
- 支援彩色終端輸出
- 動態數據更新
- 多種圖表類型
- 自適應列寬
- 實時統計計算

作者: FastAPI Demo Project
版本: 1.0.0
創建日期: 2024
"""

import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


# 顏色代碼定義
class Colors:
    """終端顏色代碼"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

@dataclass
class SystemInfo:
    """系統信息數據類"""
    platform: str
    python_version: str
    cpu_count: int
    memory_total: int
    uptime: float

@dataclass
class DataPoint:
    """數據點結構"""
    timestamp: datetime
    value: float
    category: str
    metadata: Dict[str, Any]

class ComplexOutputSystem:
    """複雜輸出系統主類"""

    def __init__(self):
        self.data_points: List[DataPoint] = []
        self.categories = ['CPU', 'Memory', 'Network', 'Storage', 'Database']
        self.colors = [Colors.RED, Colors.GREEN, Colors.YELLOW, Colors.BLUE, Colors.MAGENTA]

    def generate_sample_data(self, count: int = 100) -> None:
        """生成樣本數據"""
        print(f"{Colors.CYAN}正在生成 {count} 個樣本數據點...{Colors.END}")

        for i in range(count):
            timestamp = datetime.now() - timedelta(minutes=random.randint(0, 1440))
            value = random.uniform(0, 100)
            category = random.choice(self.categories)
            metadata = {
                'region': random.choice(['台北', '高雄', '台中', '台南']),
                'severity': random.choice(['低', '中', '高', '緊急']),
                'source': f'server_{random.randint(1, 10)}'
            }

            self.data_points.append(DataPoint(timestamp, value, category, metadata))

        print(f"{Colors.GREEN}✓ 數據生成完成{Colors.END}")

    def display_system_info(self) -> None:
        """顯示系統信息"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}系統信息概覽{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")

        info = SystemInfo(
            platform=platform.system(),
            python_version=platform.python_version(),
            cpu_count=os.cpu_count() or 1,
            memory_total=os.getenv('MEMORY_TOTAL', 'Unknown'),
            uptime=time.time()
        )

        info_data = [
            ["項目", "值", "狀態"],
            ["-" * 20, "-" * 30, "-" * 10],
            ["作業系統", info.platform, "✓"],
            ["Python 版本", info.python_version, "✓"],
            ["CPU 核心數", str(info.cpu_count), "✓"],
            ["記憶體總量", str(info.memory_total), "✓"],
            ["運行時間", f"{info.uptime:.2f} 秒", "✓"]
        ]

        self._print_table(info_data)

    def display_statistics(self) -> None:
        """顯示統計信息"""
        print(f"\n{Colors.BOLD}{Colors.YELLOW}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.YELLOW}數據統計分析{Colors.END}")
        print(f"{Colors.BOLD}{Colors.YELLOW}{'='*60}{Colors.END}")

        if not self.data_points:
            print(f"{Colors.RED}沒有數據可供分析{Colors.END}")
            return

        # 按類別統計
        category_stats = defaultdict(list)
        for point in self.data_points:
            category_stats[point.category].append(point.value)

        stats_data = [["類別", "數量", "平均值", "最大值", "最小值", "標準差"]]
        stats_data.append(["-" * 10, "-" * 8, "-" * 10, "-" * 10, "-" * 10, "-" * 10])

        for category, values in category_stats.items():
            count = len(values)
            mean_val = sum(values) / count
            max_val = max(values)
            min_val = min(values)
            std_dev = math.sqrt(sum((x - mean_val) ** 2 for x in values) / count)

            stats_data.append([
                category,
                str(count),
                f"{mean_val:.2f}",
                f"{max_val:.2f}",
                f"{min_val:.2f}",
                f"{std_dev:.2f}"
            ])

        self._print_table(stats_data)

    def display_charts(self) -> None:
        """顯示 ASCII 圖表"""
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}數據可視化圖表{Colors.END}")
        print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*60}{Colors.END}")

        # 生成直方圖
        self._display_histogram()

        # 生成折線圖
        self._display_line_chart()

        # 生成圓餅圖
        self._display_pie_chart()

    def _display_histogram(self) -> None:
        """顯示直方圖"""
        print(f"\n{Colors.CYAN}數據分布直方圖:{Colors.END}")

        # 將數據分為 10 個區間
        bins = [0] * 10
        for point in self.data_points:
            bin_index = min(int(point.value / 10), 9)
            bins[bin_index] += 1

        max_count = max(bins) if bins else 1

        for i, count in enumerate(bins):
            bar_length = int((count / max_count) * 50)
            bar = "█" * bar_length
            range_label = f"{i*10:2d}-{(i+1)*10-1:2d}"
            print(f"{range_label:>6} |{Colors.GREEN}{bar:<50}{Colors.END} {count:>3}")

    def _display_line_chart(self) -> None:
        """顯示折線圖"""
        print(f"\n{Colors.CYAN}時間序列折線圖:{Colors.END}")

        # 按時間排序並取最近 20 個點
        sorted_points = sorted(self.data_points, key=lambda x: x.timestamp)[-20:]

        if not sorted_points:
            return

        values = [point.value for point in sorted_points]
        min_val = min(values)
        max_val = max(values)

        # 正規化到 0-20 範圍
        normalized = [(v - min_val) / (max_val - min_val) * 20 if max_val > min_val else 10 for v in values]

        print("時間序列數據:")
        for i, (point, norm_val) in enumerate(zip(sorted_points, normalized)):
            bar_length = int(norm_val)
            bar = "█" * bar_length
            time_str = point.timestamp.strftime("%H:%M")
            print(f"{time_str:>5} |{Colors.BLUE}{bar:<20}{Colors.END} {point.value:>6.1f}")

    def _display_pie_chart(self) -> None:
        """顯示圓餅圖（ASCII 版本）"""
        print(f"\n{Colors.CYAN}類別分布圓餅圖:{Colors.END}")

        category_counts = Counter(point.category for point in self.data_points)
        total = sum(category_counts.values())

        if total == 0:
            return

        # 使用不同字符表示不同類別
        pie_chars = ['●', '◆', '▲', '■', '★']

        print("類別分布:")
        for i, (category, count) in enumerate(category_counts.most_common()):
            percentage = (count / total) * 100
            char = pie_chars[i % len(pie_chars)]
            color = self.colors[i % len(self.colors)]
            print(f"{color}{char} {category:<10}{Colors.END} {percentage:>5.1f}% ({count:>3} 項)")

    def display_real_time_data(self) -> None:
        """顯示實時數據流"""
        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}實時數據流監控{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}")

        print("正在模擬實時數據更新...")

        for i in range(10):
            # 生成新的數據點
            new_point = DataPoint(
                timestamp=datetime.now(),
                value=random.uniform(0, 100),
                category=random.choice(self.categories),
                metadata={'iteration': i + 1}
            )
            self.data_points.append(new_point)

            # 顯示進度條
            progress = (i + 1) / 10
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)

            print(f"\r{Colors.CYAN}進度: [{bar}] {progress*100:>5.1f}%{Colors.END}", end="", flush=True)
            time.sleep(0.5)

        print(f"\n{Colors.GREEN}✓ 實時數據更新完成{Colors.END}")

    def display_json_output(self) -> None:
        """顯示 JSON 格式輸出"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}JSON 格式數據輸出{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")

        # 準備 JSON 數據
        json_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_points": len(self.data_points),
                "categories": list(set(point.category for point in self.data_points))
            },
            "summary": self._calculate_summary(),
            "recent_data": [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "value": point.value,
                    "category": point.category,
                    "metadata": point.metadata
                }
                for point in self.data_points[-5:]  # 最近 5 個點
            ]
        }

        # 格式化輸出 JSON
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        print(json_str)

    def _calculate_summary(self) -> Dict[str, Any]:
        """計算數據摘要"""
        if not self.data_points:
            return {}

        values = [point.value for point in self.data_points]
        return {
            "total_points": len(self.data_points),
            "average_value": sum(values) / len(values),
            "min_value": min(values),
            "max_value": max(values),
            "value_range": max(values) - min(values)
        }

    def _print_table(self, data: List[List[str]]) -> None:
        """打印表格"""
        if not data:
            return

        # 計算每列的最大寬度
        col_widths = []
        for col in range(len(data[0])):
            max_width = max(len(str(row[col])) for row in data)
            col_widths.append(max_width + 2)

        # 打印表格
        for i, row in enumerate(data):
            if i == 1:  # 分隔線
                print("+" + "+".join("-" * width for width in col_widths) + "+")
            else:
                formatted_row = "|".join(f" {str(cell):<{col_widths[j]-1}}" for j, cell in enumerate(row))
                print(f"|{formatted_row}|")

    def display_progress_animation(self) -> None:
        """顯示進度動畫"""
        print(f"\n{Colors.BOLD}{Colors.YELLOW}系統初始化中...{Colors.END}")

        stages = [
            "載入配置檔案",
            "初始化資料庫連接",
            "載入業務邏輯模組",
            "啟動 API 服務",
            "註冊路由端點",
            "啟動監控服務",
            "系統就緒"
        ]

        for i, stage in enumerate(stages):
            # 進度條
            progress = (i + 1) / len(stages)
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)

            # 動畫效果
            spinner = "|/-\\"[i % 4]

            print(f"\r{Colors.CYAN}{spinner} {stage:<20} [{bar}] {progress*100:>5.1f}%{Colors.END}", end="", flush=True)
            time.sleep(0.8)

        print(f"\n{Colors.GREEN}✓ 系統初始化完成！{Colors.END}")

    def run_complex_demo(self) -> None:
        """運行複雜輸出演示"""
        print(f"{Colors.BOLD}{Colors.MAGENTA}")
        print("╔" + "═" * 58 + "╗")
        print("║" + " " * 20 + "複雜輸出系統演示" + " " * 20 + "║")
        print("╚" + "═" * 58 + "╝")
        print(f"{Colors.END}")

        # 顯示進度動畫
        self.display_progress_animation()

        # 生成樣本數據
        self.generate_sample_data(150)

        # 顯示系統信息
        self.display_system_info()

        # 顯示統計分析
        self.display_statistics()

        # 顯示圖表
        self.display_charts()

        # 顯示實時數據
        self.display_real_time_data()

        # 顯示 JSON 輸出
        self.display_json_output()

        # 最終摘要
        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}演示完成摘要{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}")

        summary = self._calculate_summary()
        print(f"總數據點數: {Colors.CYAN}{summary.get('total_points', 0)}{Colors.END}")
        print(f"平均數值: {Colors.CYAN}{summary.get('average_value', 0):.2f}{Colors.END}")
        print(f"數值範圍: {Colors.CYAN}{summary.get('min_value', 0):.2f} - {summary.get('max_value', 0):.2f}{Colors.END}")
        print(f"數據類別: {Colors.CYAN}{len(set(point.category for point in self.data_points))}{Colors.END}")

        print(f"\n{Colors.BOLD}{Colors.MAGENTA}感謝使用複雜輸出系統！{Colors.END}")

def dev():
    """
    開發環境啟動函數

    這個函數會啟動一個複雜的輸出演示系統，展示多種數據格式、
    圖表、統計分析和實時監控功能。

    功能包括：
    - 系統信息展示
    - 數據統計分析
    - ASCII 圖表繪製
    - 實時數據流模擬
    - JSON 格式輸出
    - 進度動畫效果
    """
    # 創建複雜輸出系統實例
    output_system = ComplexOutputSystem()

    # 運行複雜演示
    output_system.run_complex_demo()

    # 原始開發伺服器代碼（已註解）
    # 調整成你的 app 模組路徑，例如 fastapi_demo_02.app:app
    # uvicorn_run("fastapi_demo_02.app:app", reload=True, host="127.0.0.1", port=8000)