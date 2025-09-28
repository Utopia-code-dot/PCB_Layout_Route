import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Rectangle
import heapq
import os
import shutil
from datetime import datetime


class PCBGenerator:
    def __init__(self, pcb_size, main_max_comp_size, main_min_comp_size, sub_max_comp_size, sub_min_comp_size, num_sub_comps=10):
        """Initialize PCB generator with enhanced routing capabilities"""
        self.pcb_size = pcb_size  # (width, height) of PCB
        self.main_max_comp_size = main_max_comp_size  # (max_w, max_h) for main component
        self.main_min_comp_size = main_min_comp_size  # (min_w, min_h) for main component
        self.sub_max_comp_size = sub_max_comp_size    # (max_w, max_h) for sub components
        self.sub_min_comp_size = sub_min_comp_size    # (min_w, min_h) for sub components
        self.num_sub_comps = num_sub_comps  # Number of sub components (matches main component pins)

        # Generated PCB data
        self.pcb_data = None

        # Track occupied grid positions, wire paths and pin positions
        self.occupied_grid = set()  # 器件占据的网格
        self.wire_paths = set()     # 布线占据的网格
        self.pin_positions = set()  # 所有pin脚的位置

        # 记录主器件每个pin脚所在的边
        self.pin_edges = {}  # format: {pin_name: edge} where edge is 'left', 'right', 'top', 'bottom'

    def generate_main_component(self):
        """生成主器件，确保四周留有足够布线空间"""
        # 随机生成主器件尺寸（原逻辑保留）
        width = random.randint(self.main_min_comp_size[0], self.main_max_comp_size[0])
        height = random.randint(self.main_min_comp_size[1], self.main_max_comp_size[1])

        # 增大边距确保周围有足够布线空间
        margin = 8
        max_x = self.pcb_size[0] - width - margin
        max_y = self.pcb_size[1] - height - margin
        min_x = margin
        min_y = margin

        # 确保主器件位置在PCB中心区域，四周留有充足空间
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)

        # 记录主器件占据的网格 (不包括pin脚位置)
        for i in range(x, x + width + 1):
            for j in range(y, y + height + 1):
                self.occupied_grid.add((i, j))

        # 为主器件生成指定数量的均匀分布在四条边上的pin脚
        pins = {}
        total_pins = self.num_sub_comps

        # 计算每条边放置多少个pin脚 (尽可能均匀)
        edge_pin_counts = self._get_edge_pin_counts(total_pins)  # [bottom, right, top, left]

        # 为每条边生成pin脚
        pin_index = 1

        # 边1: 底边 (bottom)
        if edge_pin_counts[0] > 0:
            step = width / (edge_pin_counts[0] + 1) if edge_pin_counts[0] > 1 else width
            for i in range(1, edge_pin_counts[0] + 1):
                pin_x = x + int(i * step)
                pin_y = y + 1  # y = 下边界 + 1
                pin_x = max(x, min(pin_x, x + width - 1))

                if (pin_x, pin_y) not in self.pin_positions:
                    self.pin_positions.add((pin_x, pin_y))
                    self.pin_edges[f"pin{pin_index}"] = "bottom"
                    if (pin_x, pin_y) in self.occupied_grid:
                        self.occupied_grid.remove((pin_x, pin_y))
                    pins[f"pin{pin_index}"] = [pin_x - x, pin_y - y]
                    pin_index += 1

        # 边2: 右边 (right)
        if edge_pin_counts[1] > 0:
            step = height / (edge_pin_counts[1] + 1) if edge_pin_counts[1] > 1 else height
            for i in range(1, edge_pin_counts[1] + 1):
                pin_x = x + width - 1
                pin_y = y + int(i * step)
                pin_y = max(y + 1, min(pin_y, y + height))

                if (pin_x, pin_y) not in self.pin_positions:
                    self.pin_positions.add((pin_x, pin_y))
                    self.pin_edges[f"pin{pin_index}"] = "right"
                    if (pin_x, pin_y) in self.occupied_grid:
                        self.occupied_grid.remove((pin_x, pin_y))
                    pins[f"pin{pin_index}"] = [pin_x - x, pin_y - y]
                    pin_index += 1

        # 边3: 顶边 (top)
        if edge_pin_counts[2] > 0:
            step = width / (edge_pin_counts[2] + 1) if edge_pin_counts[2] > 1 else width
            for i in range(1, edge_pin_counts[2] + 1):
                pin_x = x + int(i * step)
                pin_y = y + height
                pin_x = max(x, min(pin_x, x + width - 1))

                if (pin_x, pin_y) not in self.pin_positions:
                    self.pin_positions.add((pin_x, pin_y))
                    self.pin_edges[f"pin{pin_index}"] = "top"
                    if (pin_x, pin_y) in self.occupied_grid:
                        self.occupied_grid.remove((pin_x, pin_y))
                    pins[f"pin{pin_index}"] = [pin_x - x, pin_y - y]
                    pin_index += 1

        # 边4: 左边 (left)
        if edge_pin_counts[3] > 0:
            step = height / (edge_pin_counts[3] + 1) if edge_pin_counts[3] > 1 else height
            for i in range(1, edge_pin_counts[3] + 1):
                pin_x = x
                pin_y = y + int(i * step)
                pin_y = max(y + 1, min(pin_y, y + height))

                if (pin_x, pin_y) not in self.pin_positions:
                    self.pin_positions.add((pin_x, pin_y))
                    self.pin_edges[f"pin{pin_index}"] = "left"
                    if (pin_x, pin_y) in self.occupied_grid:
                        self.occupied_grid.remove((pin_x, pin_y))
                    pins[f"pin{pin_index}"] = [pin_x - x, pin_y - y]
                    pin_index += 1

        # 确保生成足够的pin脚（补全逻辑）
        while pin_index <= total_pins:
            placed = False
            attempts = 0
            while not placed and attempts < 50:
                attempts += 1
                edge = random.randint(1, 4)
                edge_name = ["bottom", "right", "top", "left"][edge - 1]

                if edge == 1:  # 底边
                    pin_x = random.randint(x, x + width - 1)
                    pin_y = y + 1
                elif edge == 2:  # 右边
                    pin_x = x + width - 1
                    pin_y = random.randint(y + 1, y + height)
                elif edge == 3:  # 顶边
                    pin_x = random.randint(x, x + width - 1)
                    pin_y = y + height
                else:  # 左边
                    pin_x = x
                    pin_y = random.randint(y + 1, y + height)

                if (pin_x, pin_y) not in self.pin_positions:
                    self.pin_positions.add((pin_x, pin_y))
                    self.pin_edges[f"pin{pin_index}"] = edge_name
                    if (pin_x, pin_y) in self.occupied_grid:
                        self.occupied_grid.remove((pin_x, pin_y))
                    pins[f"pin{pin_index}"] = [pin_x - x, pin_y - y]
                    pin_index += 1
                    placed = True

        if len(pins) != total_pins:
            raise Exception(f"Could not generate required {total_pins} pins for main component")

        return {
            "bottom_left_position": [x, y],
            "contour_size": [width, height],
            "pins": pins,
            "pin_edges": self.pin_edges
        }

    def _get_edge_pin_counts(self, total_pins):
        """计算每条边应分配的pin脚数量，尽量均匀"""
        base = total_pins // 4
        remainder = total_pins % 4

        counts = [base, base, base, base]
        for i in range(remainder):
            counts[i] += 1

        return counts

    def generate_sub_components(self, main_comp):
        """优化子器件布局，确保留有布线通道（子器件大小随机）"""
        sub_comps = []
        main_x, main_y = main_comp["bottom_left_position"]
        main_width, main_height = main_comp["contour_size"]
        main_pins = main_comp["pins"]
        pin_edges = main_comp["pin_edges"]

        spacing = 2  # 子器件间距，为布线留出空间

        # 按边分组pin脚（后续按边布局子器件）
        edge_pins = {
            "left": [],
            "right": [],
            "top": [],
            "bottom": []
        }
        for pin_name in main_pins.keys():
            edge = pin_edges[pin_name]
            edge_pins[edge].append(pin_name)

        # 为每条边的子器件生成规整位置（子器件大小随机）
        for edge, pins in edge_pins.items():
            if not pins:
                continue

            # 1. 按pin脚位置排序（确保子器件与pin脚顺序对应）
            sorted_pins = self._sort_pins_by_position(edge, pins, main_comp)
            num_subs = len(sorted_pins)

            # 2. 为每个子器件生成随机大小（在sub_min和sub_max之间）
            sub_sizes = []  # 存储 (width, height)，顺序与sorted_pins一致
            for _ in range(num_subs):
                # 宽度：sub_min_comp_size[0] ~ sub_max_comp_size[0]
                sub_w = random.randint(self.sub_min_comp_size[0], self.sub_max_comp_size[0])
                # 高度：sub_min_comp_size[1] ~ sub_max_comp_size[1]
                sub_h = random.randint(self.sub_min_comp_size[1], self.sub_max_comp_size[1])
                sub_sizes.append((sub_w, sub_h))

            # 3. 计算布局关键参数（总长度+最大尺寸，用于确定起始位置）
            if edge in ["left", "right"]:  # 垂直排列（子器件上下堆叠）
                # 总长度 = 所有子器件高度之和 + 间距*(数量-1)
                total_length = sum(h for _, h in sub_sizes) + (num_subs - 1) * spacing
                # 最大宽度 = 所有子器件宽度的最大值（确保水平方向不超出）
                max_width = max(w for w, _ in sub_sizes)
            else:  # top/bottom（水平排列，子器件左右排列）
                # 总长度 = 所有子器件宽度之和 + 间距*(数量-1)
                total_length = sum(w for w, _ in sub_sizes) + (num_subs - 1) * spacing
                # 最大高度 = 所有子器件高度的最大值（确保垂直方向不超出）
                max_height = max(h for _, h in sub_sizes)

            # 4. 计算子器件起始位置（基于边的位置和布局参数）
            start_x, start_y = self._calculate_sub_comp_start_position(
                edge, main_x, main_y, main_width, main_height,
                total_length, max_width if edge in ["left", "right"] else max_height
            )

            # 5. 为每个子器件生成具体位置并记录
            current_offset = 0  # 记录当前子器件的偏移量（垂直/水平方向）
            for pin_name, (sub_w, sub_h) in zip(sorted_pins, sub_sizes):
                # 根据排列方向计算当前子器件的坐标
                if edge in ["left", "right"]:  # 垂直排列：x固定，y随偏移量变化
                    x = start_x
                    y = start_y + current_offset
                    # 更新偏移量（下一个子器件 = 当前高度 + 间距）
                    current_offset += sub_h + spacing
                else:  # 水平排列：y固定，x随偏移量变化
                    x = start_x + current_offset
                    y = start_y
                    # 更新偏移量（下一个子器件 = 当前宽度 + 间距）
                    current_offset += sub_w + spacing

                # 6. 确保子器件在PCB范围内（边界调整）
                if not self._is_inside_pcb(x, y, sub_w, sub_h):
                    x = max(1, min(x, self.pcb_size[0] - sub_w - 1))
                    y = max(1, min(y, self.pcb_size[1] - sub_h - 1))

                # 7. 检查位置有效性（包括周围布线空间）
                valid_position = True
                for dx in range(x - 1, x + sub_w + 2):  # 额外检查周围1格空间
                    for dy in range(y - 1, y + sub_h + 2):
                        if (dx, dy) in self.occupied_grid:
                            valid_position = False
                            break
                    if not valid_position:
                        break

                # 位置无效时，寻找附近可用位置
                if not valid_position:
                    x, y = self._find_nearby_position(x, y, sub_w, sub_h)
                    if x is None:
                        print(f"Warning: Could not find valid position for sub-component {pin_name}")
                        continue

                # 8. 记录子器件占据的网格（包括本体）
                for dx in range(sub_w + 1):
                    for dy in range(sub_h + 1):
                        self.occupied_grid.add((x + dx, y + dy))

                # 9. 生成子器件pin脚（基于当前子器件的实际大小）
                pin_position = self._generate_sub_comp_pin(edge, x, y, sub_w, sub_h)
                if pin_position is None:
                    print(f"Warning: Could not generate pin for sub-component {pin_name}")
                    continue

                # 10. 记录pin脚位置（从占据网格中移除，避免冲突）
                self.pin_positions.add(pin_position)
                if pin_position in self.occupied_grid:
                    self.occupied_grid.remove(pin_position)

                # 11. 整理子器件信息
                sub_comp_num = int(pin_name.replace("pin", ""))
                sub_comp_name = f"sub_component_{sub_comp_num}"
                sub_comps.append({
                    sub_comp_name: {
                        "bottom_left_position": [x, y],
                        "contour_size": [sub_w, sub_h],  # 保存随机生成的大小
                        "pins": {
                            "pin1": [pin_position[0] - x, pin_position[1] - y]  # pin脚相对坐标
                        },
                        "connected_main_pin": pin_name
                    }
                })

        # 按子器件编号排序（确保一致性）
        sub_comps.sort(key=lambda x: int(list(x.keys())[0].replace("sub_component_", "")))
        return sub_comps

    def _sort_pins_by_position(self, edge, pins, main_comp):
        """按pin脚在主器件边上的位置排序（确保子器件布局顺序合理）"""
        main_x, main_y = main_comp["bottom_left_position"]
        main_width, main_height = main_comp["contour_size"]
        main_pins = main_comp["pins"]

        # 计算每个pin脚的绝对坐标
        pin_positions = {}
        for pin_name in pins:
            rel_x, rel_y = main_pins[pin_name]
            abs_x = main_x + rel_x
            abs_y = main_y + rel_y
            pin_positions[pin_name] = (abs_x, abs_y)

        # 按边的方向排序：底边/顶边按x排序，左边/右边按y排序
        if edge in ["bottom", "top"]:
            return sorted(pins, key=lambda p: pin_positions[p][0])  # 水平方向排序
        else:
            return sorted(pins, key=lambda p: pin_positions[p][1])  # 垂直方向排序

    def _calculate_sub_comp_start_position(self, edge, main_x, main_y, main_width, main_height,
                                           total_length, max_dim):
        """计算子器件起始位置（适配随机大小，确保远离主器件且居中）"""
        margin = 3  # 主器件与子器件之间的最小间距（布线空间）

        if edge == "left":
            # 垂直排列（左側）：x = 主器件左边界 - 间距 - 最大子器件宽度
            x = main_x - margin - max_dim
            # y = 主器件y + (主器件高度 - 子器件总长度)/2（垂直居中）
            y = main_y + (main_height - total_length) // 2
            y = max(1, y)  # 确保不超出PCB上边界
            return x, y

        elif edge == "right":
            # 垂直排列（右側）：x = 主器件右边界 + 间距
            x = main_x + main_width + margin
            # y = 主器件y + (主器件高度 - 子器件总长度)/2（垂直居中）
            y = main_y + (main_height - total_length) // 2
            y = max(1, y)
            return x, y

        elif edge == "top":
            # 水平排列（上方）：y = 主器件上边界 + 间距
            y = main_y + main_height + margin
            # x = 主器件x + (主器件宽度 - 子器件总长度)/2（水平居中）
            x = main_x + (main_width - total_length) // 2
            x = max(1, x)  # 确保不超出PCB左边界
            return x, y

        else:  # bottom（下方）
            # 水平排列（下方）：y = 主器件下边界 - 间距 - 最大子器件高度
            y = main_y - margin - max_dim
            # x = 主器件x + (主器件宽度 - 子器件总长度)/2（水平居中）
            x = main_x + (main_width - total_length) // 2
            x = max(1, x)
            return x, y

    def _is_inside_pcb(self, x, y, width, height):
        """检查子器件是否完全在PCB范围内（含边界预留）"""
        return (x >= 1 and
                y >= 1 and
                x + width <= self.pcb_size[0] - 1 and
                y + height <= self.pcb_size[1] - 1)

    def _find_nearby_position(self, x, y, width, height, max_attempts=30):
        """螺旋式搜索附近可用位置（适配随机子器件大小）"""
        for layer in range(1, max_attempts + 1):
            # 右移搜索
            for dx in range(layer):
                new_x = x + dx + 1
                new_y = y + layer
                if self._is_valid_position(new_x, new_y, width, height):
                    return new_x, new_y
            # 上移搜索
            for dy in range(layer):
                new_x = x + layer
                new_y = y + layer - dy - 1
                if self._is_valid_position(new_x, new_y, width, height):
                    return new_x, new_y
            # 左移搜索
            for dx in range(layer):
                new_x = x + layer - dx - 1
                new_y = y - 1
                if self._is_valid_position(new_x, new_y, width, height):
                    return new_x, new_y
            # 下移搜索
            for dy in range(layer):
                new_x = x - 1
                new_y = y - 1 + dy + 1
                if self._is_valid_position(new_x, new_y, width, height):
                    return new_x, new_y
        return None, None

    def _is_valid_position(self, x, y, width, height):
        """检查位置是否有效（不重叠+在PCB内）"""
        if not self._is_inside_pcb(x, y, width, height):
            return False
        # 检查子器件本体是否与已占据网格重叠
        for dx in range(x, x + width + 1):
            for dy in range(y, y + height + 1):
                if (dx, dy) in self.occupied_grid:
                    return False
        return True

    def _generate_sub_comp_pin(self, edge, x, y, width, height):
        """基于子器件实际大小和边的位置生成pin脚（严格限定范围）"""
        if edge == "left":
            # 左側子器件：pin脚在右边界（x = x+width-1），y ∈ [y+1, y+height]
            pin_x = x + width - 1
            pin_y = random.randint(y + 1, y + height)
        elif edge == "right":
            # 右側子器件：pin脚在左边界（x = x），y ∈ [y+1, y+height]
            pin_x = x
            pin_y = random.randint(y + 1, y + height)
        elif edge == "top":
            # 上側子器件：pin脚在下边界（y = y+1），x ∈ [x, x+width-1]
            pin_y = y + 1
            pin_x = random.randint(x, x + width - 1)
        else:  # bottom（下側）
            # 下側子器件：pin脚在上边界（y = y+height），x ∈ [x, x+width-1]
            pin_y = y + height
            pin_x = random.randint(x, x + width - 1)

        # 确保pin脚不重复
        if (pin_x, pin_y) not in self.pin_positions:
            return (pin_x, pin_y)

        # 尝试同一范围内的其他位置（最多10次）
        attempts = 0
        max_attempts = 10
        while attempts < max_attempts:
            attempts += 1
            if edge in ["left", "right"]:
                new_pin_y = random.randint(y + 1, y + height)
                new_pin_x = pin_x
            else:
                new_pin_x = random.randint(x, x + width - 1)
                new_pin_y = pin_y
            if (new_pin_x, new_pin_y) not in self.pin_positions:
                return (new_pin_x, new_pin_y)
        return None

    def define_net_connections(self):
        """定义网络连接关系（主器件pin脚与对应子器件pin脚连接）"""
        net_info = {}
        for i in range(1, self.num_sub_comps + 1):
            net_info[f"net{i}"] = {
                "main_component": f"pin{i}",
                f"sub_component_{i}": "pin1"
            }
        return net_info

    def a_star_algorithm(self, start, end):
        """增强版A*算法（适配随机子器件大小的布线）"""
        # 四方向优先（减少路径复杂度），八方向备用
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # 曼哈顿距离启发函数（适合网格布线）
        def heuristic(node, goal):
            return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

        # 初始化优先队列和路径记录
        open_heap = []
        heapq.heappush(open_heap, (0, start))
        came_from = {start: None}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}

        max_iterations = 50000  # 增加迭代上限，适配复杂布局
        iteration = 0

        while open_heap and iteration < max_iterations:
            iteration += 1
            _, current = heapq.heappop(open_heap)

            # 到达终点，回溯路径
            if current == end:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]  # 反转路径（从起点到终点）

            # 探索邻居节点
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                # 检查是否在PCB边界内
                if (neighbor[0] < 0 or neighbor[0] >= self.pcb_size[0] or
                        neighbor[1] < 0 or neighbor[1] >= self.pcb_size[1]):
                    continue

                # 允许布线靠近pin脚（但不重叠器件本体）
                if (neighbor in self.occupied_grid and
                        neighbor != start and neighbor != end and
                        not self._is_adjacent_to_pin(neighbor)):
                    continue

                # 计算移动成本（直线1.0，斜线1.414，靠近布线增加成本）
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                if neighbor in self.wire_paths:
                    move_cost += 0.5  # 避免布线过度重叠

                tentative_g_score = g_score[current] + move_cost

                # 更新更优路径
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

        return None  # 路径搜索失败（后续有 fallback 逻辑）

    def _is_adjacent_to_pin(self, position):
        """检查位置是否靠近pin脚（允许布线靠近pin脚）"""
        x, y = position
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (x + dx, y + dy) in self.pin_positions:
                    return True
        return False

    def generate_wiring(self, net_info, comp_info):
        """改进布线策略（适配随机子器件大小，确保100%布线成功率）"""
        wire_paths = {}
        main_comp = comp_info["main_component"]
        main_x, main_y = main_comp["bottom_left_position"]

        # 按边分组布线（减少不同边网络的交叉干扰）
        edge_nets = {
            "left": [], "right": [], "top": [], "bottom": []
        }
        for net_name, connections in net_info.items():
            main_pin_name = connections["main_component"]
            edge = main_comp["pin_edges"][main_pin_name]
            edge_nets[edge].append((net_name, connections))

        # 按边顺序布线（左→右→上→下）
        for edge in ["left", "right", "top", "bottom"]:
            for net_name, connections in edge_nets[edge]:
                # 获取主器件和子器件的pin脚绝对坐标
                main_pin_name = connections["main_component"]
                sub_comp_name = [name for name in connections if name.startswith("sub_component")][0]
                sub_pin_name = connections[sub_comp_name]

                sub_comp = comp_info[sub_comp_name]
                sub_x, sub_y = sub_comp["bottom_left_position"]
                sub_w, sub_h = sub_comp["contour_size"]  # 使用随机生成的子器件大小

                # 计算pin脚绝对位置
                main_pin_rel = main_comp["pins"][main_pin_name]
                main_pin_abs = (main_x + main_pin_rel[0], main_y + main_pin_rel[1])
                sub_pin_rel = sub_comp["pins"][sub_pin_name]
                sub_pin_abs = (sub_x + sub_pin_rel[0], sub_y + sub_pin_rel[1])

                # 多次尝试布线（最多10次）
                path = None
                attempts = 0
                max_wire_attempts = 10
                current_wire_paths = self.wire_paths.copy()  # 保存当前布线状态

                while path is None and attempts < max_wire_attempts:
                    attempts += 1
                    self.wire_paths = current_wire_paths.copy()  # 恢复初始状态
                    path = self.a_star_algorithm(main_pin_abs, sub_pin_abs)

                    # 最后几次尝试放宽障碍限制（允许靠近器件）
                    if path is None and attempts >= max_wire_attempts // 2:
                        self.occupied_grid = self.occupied_grid - self._get_near_pin_positions()

                # 仍失败时，使用 fallback 策略（强制布线）
                if path is None:
                    print(f"Using fallback routing for {net_name}")
                    self.occupied_grid = self.occupied_grid - self._get_near_pin_positions()
                    path = self.a_star_algorithm(main_pin_abs, sub_pin_abs)

                # 终极 fallback：直线连接
                if path is None:
                    print(f"Using direct line for {net_name}")
                    path = self._direct_line_fallback(main_pin_abs, sub_pin_abs)

                # 记录布线路径
                if path:
                    for point in path:
                        self.wire_paths.add(point)
                    wire_paths[net_name] = path

        return wire_paths

    def _get_near_pin_positions(self):
        """获取pin脚附近区域（允许布线通过）"""
        near_pins = set()
        for (x, y) in self.pin_positions:
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    near_pins.add((x + dx, y + dy))
        return near_pins

    def _direct_line_fallback(self, start, end):
        """直线连接作为最后的布线方案（适配随机子器件大小）"""
        path = []
        x0, y0 = start
        x1, y1 = end

        #  Bresenham 直线算法（生成直线上的所有点）
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        err = dx - dy

        while True:
            path.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        # 强制添加布线（即使靠近器件）
        for point in path:
            self.wire_paths.add(point)
            if point in self.occupied_grid:
                self.occupied_grid.remove(point)

        return path

    def generate_pcb(self):
        """生成完整的PCB数据（适配随机子器件大小，确保100%布线成功率）"""
        # 重置状态（避免多次生成时的状态污染）
        self.occupied_grid = set()
        self.wire_paths = set()
        self.pin_positions = set()
        self.pin_edges = {}

        # 1. 生成主器件（原逻辑保留）
        main_comp = self.generate_main_component()

        # 2. 生成子器件（核心改动：子器件大小随机）
        sub_comps = self.generate_sub_components(main_comp)

        # 3. 整合器件信息
        comp_info = {"main_component": main_comp}
        for sub_comp in sub_comps:
            comp_info.update(sub_comp)

        # 4. 定义网络连接
        net_info = self.define_net_connections()

        # 5. 布线（适配随机子器件布局）
        wire_paths = self.generate_wiring(net_info, comp_info)

        # 6. 验证布线完整性（失败则重试）
        if len(wire_paths) < len(net_info):
            print(f"Retrying PCB generation (missing {len(net_info)-len(wire_paths)} wires)")
            return self.generate_pcb()

        # 7. 组织最终数据结构
        self.pcb_data = {
            "pcb1": {
                "component_info": comp_info,
                "net_info": net_info,
                "wire_paths": wire_paths
            }
        }

        return self.pcb_data

    def save_to_json(self, folder_path, filename=None):
        """将PCB数据保存到JSON文件（适配随机子器件大小的序列化）"""
        if not self.pcb_data:
            print("请先生成PCB数据（调用 generate_pcb() 方法）")
            return

        # 确保文件夹存在
        os.makedirs(folder_path, exist_ok=True)

        # 生成唯一文件名（时间戳+随机数）
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_suffix = random.randint(1000, 9999)
            filename = f"pcb_layout_{timestamp}_{random_suffix}.json"

        full_path = os.path.join(folder_path, filename)

        # 自定义JSON编码器（处理集合和元组）
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, set):
                    return list(obj)
                if isinstance(obj, tuple):
                    return list(obj)
                return json.JSONEncoder.default(self, obj)

        # 保存数据
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(self.pcb_data, f, ensure_ascii=False, indent=2, cls=CustomEncoder)

        return full_path

    def visualize(self, title="PCB Layout and Wiring Diagram"):
        """可视化PCB布局（适配随机子器件大小的显示）"""
        if not self.pcb_data:
            print("请先生成PCB数据（调用 generate_pcb() 方法）")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # 1. 绘制PCB边界
        ax.add_patch(Rectangle((0, 0), self.pcb_size[0], self.pcb_size[1],
                               fill=False, edgecolor='black', linewidth=2))

        # 2. 获取数据
        comp_info = self.pcb_data["pcb1"]["component_info"]
        net_info = self.pcb_data["pcb1"]["net_info"]
        wire_paths = self.pcb_data["pcb1"]["wire_paths"]

        # 3. 绘制主器件
        main_comp = comp_info["main_component"]
        main_x, main_y = main_comp["bottom_left_position"]
        main_w, main_h = main_comp["contour_size"]
        ax.add_patch(Rectangle((main_x, main_y), main_w, main_h,
                               fill=True, edgecolor='blue', facecolor='lightblue', alpha=0.7))
        ax.text(main_x + main_w/2, main_y + main_h/2, "Main",
                ha='center', va='center', fontsize=10, fontweight='bold')

        # 4. 绘制主器件pin脚
        for pin_name, rel_pos in main_comp["pins"].items():
            pin_x = main_x + rel_pos[0]
            pin_y = main_y + rel_pos[1]
            ax.plot(pin_x, pin_y, 'bo', markersize=6, label="Main Pin" if "Main Pin" not in ax.get_legend_handles_labels()[1] else "")
            ax.text(pin_x + 0.3, pin_y + 0.3, pin_name, fontsize=7)

        # 5. 绘制子器件（适配随机大小）
        sub_comp_names = [name for name in comp_info if name.startswith("sub_component")]
        colors = plt.cm.Set3(np.linspace(0, 1, len(sub_comp_names)))  # 随机颜色

        for i, sub_name in enumerate(sub_comp_names):
            sub_comp = comp_info[sub_name]
            sub_x, sub_y = sub_comp["bottom_left_position"]
            sub_w, sub_h = sub_comp["contour_size"]  # 使用随机生成的大小
            # 绘制子器件矩形
            ax.add_patch(Rectangle((sub_x, sub_y), sub_w, sub_h,
                                   fill=True, edgecolor='green', facecolor=colors[i], alpha=0.7))
            # 添加子器件标签（显示编号）
            ax.text(sub_x + sub_w/2, sub_y + sub_h/2, f"S{i+1}",
                    ha='center', va='center', fontsize=8, fontweight='bold')

            # 绘制子器件pin脚
            for pin_name, rel_pos in sub_comp["pins"].items():
                pin_x = sub_x + rel_pos[0]
                pin_y = sub_y + rel_pos[1]
                ax.plot(pin_x, pin_y, 'go', markersize=6, label="Sub Pin" if "Sub Pin" not in ax.get_legend_handles_labels()[1] else "")
                ax.text(pin_x + 0.3, pin_y + 0.3, pin_name, fontsize=7)

        # 6. 绘制布线（每条网络不同颜色）
        wire_colors = plt.cm.tab10(np.linspace(0, 1, len(wire_paths)))
        for idx, (net_name, path) in enumerate(wire_paths.items()):
            if path and len(path) > 1:
                x_values = [p[0] for p in path]
                y_values = [p[1] for p in path]
                ax.plot(x_values, y_values, '-', color=wire_colors[idx], linewidth=1.5,
                        label=f"{net_name}" if idx < 10 else "")  # 限制图例数量

        # 7. 设置坐标轴和图例
        ax.set_xlim(-1, self.pcb_size[0] + 1)
        ax.set_ylim(-1, self.pcb_size[1] + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("X (Grid)", fontsize=12)
        ax.set_ylabel("Y (Grid)", fontsize=12)

        # 显示网格刻度
        ax.set_xticks(range(self.pcb_size[0] + 1))
        ax.set_yticks(range(self.pcb_size[1] + 1))

        # 调整图例位置
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        return fig

    def load_from_json(self, filename):
        """从JSON文件加载PCB数据（适配随机子器件大小的恢复）"""
        if not os.path.exists(filename):
            print(f"文件 {filename} 不存在")
            return

        # 加载数据
        with open(filename, 'r', encoding='utf-8') as f:
            self.pcb_data = json.load(f)

        # 恢复状态（占据网格、pin脚、布线）
        self.occupied_grid = set()
        self.wire_paths = set()
        self.pin_positions = set()

        comp_info = self.pcb_data["pcb1"]["component_info"]
        for comp_name, comp_data in comp_info.items():
            x, y = comp_data["bottom_left_position"]
            width, height = comp_data["contour_size"]  # 恢复随机生成的大小

            # 记录器件占据的网格
            for dx in range(width + 1):
                for dy in range(height + 1):
                    self.occupied_grid.add((x + dx, y + dy))

            # 记录pin脚位置（从占据网格中移除）
            for pin_rel in comp_data["pins"].values():
                pin_x = x + pin_rel[0]
                pin_y = y + pin_rel[1]
                self.pin_positions.add((pin_x, pin_y))
                if (pin_x, pin_y) in self.occupied_grid:
                    self.occupied_grid.remove((pin_x, pin_y))

        # 恢复布线位置
        wire_paths = self.pcb_data["pcb1"]["wire_paths"]
        for path in wire_paths.values():
            for point in path:
                self.wire_paths.add((point[0], point[1]))


def generate_encoding_table(pcb_size=30, output_folder="encode_table", filename="encode_table.json"):
    """
    生成编码表并保存为JSON文件

    编码规则:
    0:<PAD>
    1:<SOS>
    2:<EOS>
    3:<SOBS>
    4:<EOBS>
    5:<SONS>
    6:<EONS>
    7:<SODPS>
    8:<EODPS>
    9:<SEP>
    10:<SEPN>
    11:<SEPND>
    12->12+pcb_size-1:PCB板子内部的位置编码
    12+pcb_size-1+1:PCB板子轮廓大小的编码
    12+pcb_size-1+2->12+pcb_size-1+2+pcb_size-2:主器件和子器件轮廓大小的编码
    12+pcb_size-1+2+pcb_size-2+1->12+pcb_size-1+2+pcb_size-2+1+8-1:布线方向信息的编码
    """
    encode_table = {}

    # 基础符号编码
    encode_table[0] = "<PAD>"
    encode_table[1] = "<SOS>"
    encode_table[2] = "<EOS>"
    encode_table[3] = "<SOBS>"
    encode_table[4] = "<EOBS>"
    encode_table[5] = "<SONS>"
    encode_table[6] = "<EONS>"
    encode_table[7] = "<SODPS>"
    encode_table[8] = "<EODPS>"
    encode_table[9] = "<SEP>"
    encode_table[10] = "<SEPN>"
    encode_table[11] = "<SEPND>"

    # PCB板子内部的位置编码 (12 到 12 + pcb_size - 1)
    position_start = 12
    position_end = 12 + pcb_size - 1
    for code in range(position_start, position_end + 1):
        # 计算对应的坐标值 (从0开始)
        coord = code - position_start
        encode_table[code] = f"POS_{coord}"

    # PCB板子轮廓大小的编码 (12 + pcb_size - 1 + 1)
    board_size_code = position_end + 1
    encode_table[board_size_code] = "BOARD_SIZE"

    # 主器件和子器件轮廓大小的编码
    # (12 + pcb_size - 1 + 2 到 12 + pcb_size - 1 + 2 + pcb_size - 2)
    component_size_start = board_size_code + 1
    component_size_end = component_size_start + pcb_size - 2
    for code in range(component_size_start, component_size_end + 1):
        size = code - component_size_start + 1  # 大小从1开始
        encode_table[code] = f"COMP_SIZE_{size}"

    # 布线方向信息的编码 (8个方向)
    # (12 + pcb_size - 1 + 2 + pcb_size - 2 + 1 到 ... + 8 - 1)
    direction_start = component_size_end + 1
    directions = [
        "DIR_UP", "DIR_DOWN", "DIR_LEFT", "DIR_RIGHT",
        "DIR_UP_LEFT", "DIR_UP_RIGHT", "DIR_DOWN_LEFT", "DIR_DOWN_RIGHT"
    ]
    for i, direction in enumerate(directions):
        encode_table[direction_start + i] = direction

    # 确保文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 保存编码表到JSON文件
    full_path = os.path.join(output_folder, filename)
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(encode_table, f, ensure_ascii=False, indent=2)

    print(f"编码表已生成并保存到: {os.path.abspath(full_path)}")
    print(f"编码表包含 {len(encode_table)} 个编码")
    return encode_table


def generate_multiple_datasets(num_datasets=10, output_folder="datasets",
                               visualize_all=False, visualize_samples=False, sample_count=2, split_ratio=0.9):
    """
    批量生成多个PCB数据集，按9:1划分训练集/测试集
    - 训练集: output_folder/train_data/
    - 测试集: output_folder/test_data/

    参数:
    - num_datasets: 总数据集数量（需≥10，确保测试集至少1个）
    - output_folder: 根目录（默认datasets）
    - visualize_all: 是否可视化所有生成的数据集
    - visualize_samples: 是否可视化部分样本（当visualize_all为False时有效）
    - sample_count: 要可视化的样本数量（仅当visualize_samples为True时有效）
    """
    # 1. 定义目录结构
    train_dir = os.path.join(output_folder, "train_data")  # 训练集目录
    test_dir = os.path.join(output_folder, "test_data")  # 测试集目录
    temp_dir = os.path.join(output_folder, "temp_data")  # 临时目录（生成过渡用）

    # 2. 处理目录创建与清空
    if os.path.exists(output_folder):
        # 处理训练集目录
        if os.path.exists(train_dir):
            response = input(f"训练集目录 '{train_dir}' 已存在，是否清空? (y/n): ")
            if response.lower() == 'y':
                for f in os.listdir(train_dir):
                    f_path = os.path.join(train_dir, f)
                    try:
                        if os.path.isfile(f_path):
                            os.remove(f_path)
                        elif os.path.isdir(f_path):
                            shutil.rmtree(f_path)
                    except Exception as e:
                        print(f"清空训练集文件失败: {e}")
        else:
            os.makedirs(train_dir)

        # 处理测试集目录
        if os.path.exists(test_dir):
            response = input(f"测试集目录 '{test_dir}' 已存在，是否清空? (y/n): ")
            if response.lower() == 'y':
                for f in os.listdir(test_dir):
                    f_path = os.path.join(test_dir, f)
                    try:
                        if os.path.isfile(f_path):
                            os.remove(f_path)
                        elif os.path.isdir(f_path):
                            shutil.rmtree(f_path)
                    except Exception as e:
                        print(f"清空测试集文件失败: {e}")
        else:
            os.makedirs(test_dir)

        # 处理临时目录（强制清空）
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
    else:
        # 根目录不存在，创建所有目录
        os.makedirs(output_folder)
        os.makedirs(train_dir)
        os.makedirs(test_dir)
        os.makedirs(temp_dir)

    # 4. 生成所有数据集到临时目录
    print(f"\n开始生成 {num_datasets} 个PCB数据集（临时目录: {temp_dir}）...")

    pbar = tqdm(range(num_datasets))  # 创建进度条
    pbar.set_description("正在生成数据...")
    for i in pbar:
        # 创建PCB生成器实例
        pcb_gen = PCBGenerator(
            pcb_size=(30, 30),
            main_max_comp_size=(14, 14),
            main_min_comp_size=(8, 8),
            sub_max_comp_size=(4, 4),
            sub_min_comp_size=(1, 1),
            num_sub_comps=random.randint(8, 8)
        )

        # 生成PCB数据
        pcb_data = pcb_gen.generate_pcb()
        # 保存到临时目录
        pcb_gen.save_to_json(temp_dir)

        # 可视化逻辑（可选）
        if visualize_all:
            print(f"可视化第 {i + 1} 个数据集...")
            fig = pcb_gen.visualize(title=f"PCB布局 - 第 {i + 1} 个数据集")
            plt.show()
        elif visualize_samples and i < sample_count:
            print(f"\n可视化样本 {i + 1}/{sample_count}...")
            fig = pcb_gen.visualize(title=f"PCB布局样本 {i + 1}")
            plt.show()

    # 5. 按9:1划分训练集/测试集
    # 获取临时目录中的所有JSON文件
    temp_files = [f for f in os.listdir(temp_dir) if f.endswith(".json")]
    actual_count = len(temp_files)
    if actual_count != num_datasets:
        print(f"警告：实际生成 {actual_count} 个数据集（预期 {num_datasets} 个），将按实际数量划分")
        num_datasets = actual_count

    train_size = int(num_datasets * split_ratio)
    test_size = num_datasets - train_size

    # 随机打乱文件（设置种子确保划分可复现）
    random.seed(42)
    random.shuffle(temp_files)

    # 移动文件到训练集
    print(f"\n划分数据集：训练集 {train_size} 个，测试集 {test_size} 个")
    print(f"正在移动文件到训练集目录 {train_dir}...")
    for f in temp_files[:train_size]:
        src = os.path.join(temp_dir, f)
        dst = os.path.join(train_dir, f)
        shutil.move(src, dst)

    # 移动文件到测试集
    print(f"正在移动文件到测试集目录 {test_dir}...")
    for f in temp_files[train_size:]:
        src = os.path.join(temp_dir, f)
        dst = os.path.join(test_dir, f)
        shutil.move(src, dst)

    # 6. 清理临时目录
    shutil.rmtree(temp_dir)

    # 7. 打印最终结果
    final_train_count = len(os.listdir(train_dir))
    final_test_count = len(os.listdir(test_dir))
    print(f"\n数据集生成与划分完成！")
    print(f"训练集：{final_train_count} 个文件 | 路径：{os.path.abspath(train_dir)}")
    print(f"测试集：{final_test_count} 个文件 | 路径：{os.path.abspath(test_dir)}")


if __name__ == "__main__":
    # 步骤1：生成编码表（保存到encode_table/目录）
    generate_encoding_table(pcb_size=30)

    # 步骤2：生成50个数据集，按9:1划分到datasets/train_data和datasets/test_data
    generate_multiple_datasets(
        num_datasets=10000,  # 总数据集数量（50个→训练集45个，测试集5个）
        output_folder="datasets",  # 根目录
        visualize_all=False,  # 不可视化所有
        visualize_samples=False,  # 不可视化样本（快速生成）
        sample_count=2,  # 若开启可视化，显示2个样本
        split_ratio=0.9
    )