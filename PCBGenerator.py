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
    def __init__(self, pcb_size=(30, 30), main_comp_min_size=(8, 8), main_comp_max_size=(10, 10),
                 sub_comp_size=(2, 1), num_sub_comps=10):
        """Initialize PCB generator with enhanced routing capabilities"""
        self.pcb_size = pcb_size
        self.main_comp_min_size = main_comp_min_size
        self.main_comp_max_size = main_comp_max_size
        self.sub_comp_size = sub_comp_size
        self.num_sub_comps = num_sub_comps

        # Generated PCB data
        self.pcb_data = None

        # Track occupied grid positions, wire paths and pin positions
        self.occupied_grid = set()  # 器件占据的网格
        self.wire_paths = set()  # 布线占据的网格
        self.pin_positions = set()  # 所有pin脚的位置

        # 记录主器件每个pin脚所在的边
        self.pin_edges = {}  # format: {pin_name: edge} where edge is 'left', 'right', 'top', 'bottom'

    def generate_main_component(self):
        """生成主器件，确保四周留有足够布线空间"""
        # 随机生成主器件尺寸 (8x8 到 10x10之间)
        width = random.randint(self.main_comp_min_size[0], self.main_comp_max_size[0])
        height = random.randint(self.main_comp_min_size[1], self.main_comp_max_size[1])

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

        # 为主器件生成10个均匀分布在四条边上的pin脚
        pins = {}
        total_pins = self.num_sub_comps

        # 计算每条边放置多少个pin脚 (尽可能均匀)
        edge_pin_counts = self._get_edge_pin_counts(total_pins)
        # edge_pin_counts 顺序: [bottom, right, top, left]

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

        # 确保生成足够的pin脚
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
        """优化子器件布局，确保留有布线通道"""
        sub_comps = []
        main_x, main_y = main_comp["bottom_left_position"]
        main_width, main_height = main_comp["contour_size"]
        main_pins = main_comp["pins"]
        pin_edges = main_comp["pin_edges"]

        sub_comp_width, sub_comp_height = self.sub_comp_size
        spacing = 2  # 增加子器件间距，为布线留出空间

        # 按边分组pin脚
        edge_pins = {
            "left": [],
            "right": [],
            "top": [],
            "bottom": []
        }

        for pin_name in main_pins.keys():
            edge = pin_edges[pin_name]
            edge_pins[edge].append(pin_name)

        # 为每条边的子器件生成规整位置
        for edge, pins in edge_pins.items():
            if not pins:
                continue

            sorted_pins = self._sort_pins_by_position(edge, pins, main_comp)

            # 计算该边子器件的起始位置，确保远离主器件留出布线空间
            start_x, start_y = self._calculate_sub_comp_start_position(
                edge, main_x, main_y, main_width, main_height,
                sub_comp_width, sub_comp_height, len(pins)
            )

            # 为每个pin脚生成对应的子器件
            for i, pin_name in enumerate(sorted_pins):
                if edge in ["left", "right"]:
                    x = start_x
                    y = start_y + i * (sub_comp_height + spacing)
                else:  # top, bottom
                    x = start_x + i * (sub_comp_width + spacing)
                    y = start_y

                # 确保子器件在PCB范围内
                if not self._is_inside_pcb(x, y, sub_comp_width, sub_comp_height):
                    x = max(1, min(x, self.pcb_size[0] - sub_comp_width - 1))
                    y = max(1, min(y, self.pcb_size[1] - sub_comp_height - 1))

                # 检查位置是否有效，增加布线空间检查
                valid_position = True
                # 不仅检查子器件本身，还要检查周围预留的布线空间
                for dx in range(x - 1, x + sub_comp_width + 2):
                    for dy in range(y - 1, y + sub_comp_height + 2):
                        if (dx, dy) in self.occupied_grid:
                            valid_position = False
                            break
                    if not valid_position:
                        break

                if not valid_position:
                    x, y = self._find_nearby_position(x, y, sub_comp_width, sub_comp_height)
                    if x is None:
                        print(f"Warning: Could not find valid position for sub-component {pin_name}")
                        continue

                # 记录子器件占据的网格，包括周围的保护区域
                for dx in range(sub_comp_width + 1):
                    for dy in range(sub_comp_height + 1):
                        self.occupied_grid.add((x + dx, y + dy))

                # 生成子器件pin脚
                pin_position = self._generate_sub_comp_pin(edge, x, y, sub_comp_width, sub_comp_height)

                if pin_position is None:
                    print(f"Warning: Could not generate pin for sub-component {pin_name}")
                    continue

                self.pin_positions.add(pin_position)
                if pin_position in self.occupied_grid:
                    self.occupied_grid.remove(pin_position)

                sub_comp_num = int(pin_name.replace("pin", ""))
                sub_comp_name = f"sub_component_{sub_comp_num}"

                sub_comp = {
                    sub_comp_name: {
                        "bottom_left_position": [x, y],
                        "contour_size": [sub_comp_width, sub_comp_height],
                        "pins": {
                            "pin1": [pin_position[0] - x, pin_position[1] - y]
                        },
                        "connected_main_pin": pin_name
                    }
                }
                sub_comps.append(sub_comp)

        sub_comps.sort(key=lambda x: int(list(x.keys())[0].replace("sub_component_", "")))
        return sub_comps

    def _sort_pins_by_position(self, edge, pins, main_comp):
        """按pin脚在主器件边上的位置排序"""
        main_x, main_y = main_comp["bottom_left_position"]
        main_width, main_height = main_comp["contour_size"]
        main_pins = main_comp["pins"]

        pin_positions = {}
        for pin_name in pins:
            rel_x, rel_y = main_pins[pin_name]
            abs_x = main_x + rel_x
            abs_y = main_y + rel_y
            pin_positions[pin_name] = (abs_x, abs_y)

        if edge in ["bottom", "top"]:
            return sorted(pins, key=lambda p: pin_positions[p][0])
        else:
            return sorted(pins, key=lambda p: pin_positions[p][1])

    def _calculate_sub_comp_start_position(self, edge, main_x, main_y, main_width, main_height,
                                           sub_width, sub_height, num_sub_comps):
        """优化子器件起始位置，增加与主器件的距离以留出布线空间"""
        margin = 3  # 增加边距，为主器件和子器件之间留出布线通道

        if edge == "left":
            x = main_x - sub_width - margin
            total_height = num_sub_comps * sub_height + (num_sub_comps - 1) * 2
            y = main_y + (main_height - total_height) // 2
            return x, y

        elif edge == "right":
            x = main_x + main_width + margin
            total_height = num_sub_comps * sub_height + (num_sub_comps - 1) * 2
            y = main_y + (main_height - total_height) // 2
            return x, y

        elif edge == "top":
            y = main_y + main_height + margin
            total_width = num_sub_comps * sub_width + (num_sub_comps - 1) * 2
            x = main_x + (main_width - total_width) // 2
            return x, y

        else:  # bottom
            y = main_y - sub_height - margin
            total_width = num_sub_comps * sub_width + (num_sub_comps - 1) * 2
            x = main_x + (main_width - total_width) // 2
            return x, y

    def _is_inside_pcb(self, x, y, width, height):
        """检查子器件是否在PCB范围内"""
        return (x >= 1 and
                y >= 1 and
                x + width <= self.pcb_size[0] - 1 and
                y + height <= self.pcb_size[1] - 1)

    def _find_nearby_position(self, x, y, width, height, max_attempts=30):
        """扩大搜索范围，确保找到合适的子器件位置"""
        # 螺旋式搜索，先近后远
        for layer in range(1, max_attempts + 1):
            # 右移
            for dx in range(layer):
                new_x = x + dx + 1
                new_y = y + layer
                if self._is_valid_position(new_x, new_y, width, height):
                    return new_x, new_y
            # 上移
            for dy in range(layer):
                new_x = x + layer
                new_y = y + layer - dy - 1
                if self._is_valid_position(new_x, new_y, width, height):
                    return new_x, new_y
            # 左移
            for dx in range(layer):
                new_x = x + layer - dx - 1
                new_y = y - 1
                if self._is_valid_position(new_x, new_y, width, height):
                    return new_x, new_y
            # 下移
            for dy in range(layer):
                new_x = x - 1
                new_y = y - 1 + dy + 1
                if self._is_valid_position(new_x, new_y, width, height):
                    return new_x, new_y

        return None, None

    def _is_valid_position(self, x, y, width, height):
        """检查位置是否有效，包括周围布线空间"""
        if not self._is_inside_pcb(x, y, width, height):
            return False

        for dx in range(x, x + width + 1):
            for dy in range(y, y + height + 1):
                if (dx, dy) in self.occupied_grid:
                    return False
        return True

    def _generate_sub_comp_pin(self, edge, x, y, width, height):
        """严格按照指定范围生成子器件pin脚"""
        if edge == "left":
            # 范围2：x = 器件右边界-1，y ∈ [下边界+1, 上边界]
            pin_x = x + width - 1
            pin_y = random.randint(y + 1, y + height)

        elif edge == "right":
            # 范围4：x = 器件左边界，y ∈ [下边界+1, 上边界]
            pin_x = x
            pin_y = random.randint(y + 1, y + height)

        elif edge == "top":
            # 范围1：x ∈ [左边界, 右边界-1], y = 下边界+1
            pin_y = y + 1
            pin_x = random.randint(x, x + width - 1)

        else:  # bottom
            # 范围3：x ∈ [左边界, 右边界-1], y = 上边界
            pin_y = y + height
            pin_x = random.randint(x, x + width - 1)

        # 确保pin脚位置有效
        if (pin_x, pin_y) not in self.pin_positions:
            return (pin_x, pin_y)

        # 在同一范围内尝试其他位置
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
        """定义网络连接关系"""
        net_info = {}
        for i in range(1, self.num_sub_comps + 1):
            net_info[f"net{i}"] = {
                "main_component": f"pin{i}",
                f"sub_component_{i}": "pin1"
            }
        return net_info

    def a_star_algorithm(self, start, end):
        """增强版A*算法，提高路径搜索能力"""
        # 四方向优先，减少路径复杂度
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # 使用曼哈顿距离作为启发函数，更适合网格布线
        def heuristic(node, goal):
            return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

        # 初始化开放列表和关闭列表
        open_heap = []
        heapq.heappush(open_heap, (0, start))

        came_from = {start: None}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}

        # 大幅增加最大迭代次数
        max_iterations = 50000
        iteration = 0

        while open_heap and iteration < max_iterations:
            iteration += 1
            _, current = heapq.heappop(open_heap)

            if current == end:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                # 检查是否在PCB边界内
                if (neighbor[0] < 0 or neighbor[0] >= self.pcb_size[0] or
                        neighbor[1] < 0 or neighbor[1] >= self.pcb_size[1]):
                    continue

                # 允许布线靠近但不重叠器件
                if (neighbor in self.occupied_grid and
                        neighbor != start and neighbor != end and
                        not self._is_adjacent_to_pin(neighbor)):
                    continue

                # 计算移动成本，优先直线移动
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                # 对靠近其他布线的路径增加少量成本，鼓励分散布线
                if neighbor in self.wire_paths:
                    move_cost += 0.5

                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

        return None

    def _is_adjacent_to_pin(self, position):
        """检查位置是否靠近pin脚，允许布线靠近pin脚"""
        x, y = position
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (x + dx, y + dy) in self.pin_positions:
                    return True
        return False

    def generate_wiring(self, net_info, comp_info):
        """改进布线策略，确保所有网络都能布线"""
        wire_paths = {}
        main_comp = comp_info["main_component"]
        main_x, main_y = main_comp["bottom_left_position"]

        # 按边分组布线，避免交叉干扰
        edge_nets = {
            "left": [], "right": [], "top": [], "bottom": []
        }

        # 先按边分组网络
        for net_name, connections in net_info.items():
            main_pin_name = connections["main_component"]
            edge = main_comp["pin_edges"][main_pin_name]
            edge_nets[edge].append((net_name, connections))

        # 按边顺序布线，减少不同边网络之间的干扰
        for edge in ["left", "right", "top", "bottom"]:
            for net_name, connections in edge_nets[edge]:
                main_pin_name = connections["main_component"]
                sub_comp_name = [name for name in connections if name.startswith("sub_component")][0]
                sub_pin_name = connections[sub_comp_name]

                sub_comp = comp_info[sub_comp_name]
                sub_x, sub_y = sub_comp["bottom_left_position"]

                # 计算pin脚绝对位置
                main_pin_rel = main_comp["pins"][main_pin_name]
                main_pin_abs = (main_x + main_pin_rel[0], main_y + main_pin_rel[1])

                sub_pin_rel = sub_comp["pins"][sub_pin_name]
                sub_pin_abs = (sub_x + sub_pin_rel[0], sub_y + sub_pin_rel[1])

                # 多次尝试布线，确保成功
                path = None
                attempts = 0
                max_wire_attempts = 10  # 大幅增加尝试次数

                # 保存当前布线状态，用于重试
                current_wire_paths = self.wire_paths.copy()

                while path is None and attempts < max_wire_attempts:
                    attempts += 1
                    # 每次尝试前恢复初始状态
                    self.wire_paths = current_wire_paths.copy()

                    path = self.a_star_algorithm(main_pin_abs, sub_pin_abs)

                    # 如果失败，尝试临时放宽障碍限制
                    if path is None:
                        # 最后几次尝试允许布线更靠近器件
                        self.occupied_grid = self.occupied_grid - self._get_near_pin_positions()

                if path:
                    for point in path:
                        self.wire_paths.add(point)
                    wire_paths[net_name] = path
                else:
                    # 终极方案：强制布线，即使需要靠近器件
                    print(f"Using fallback routing for {net_name}")
                    self.occupied_grid = self.occupied_grid - self._get_near_pin_positions()
                    path = self.a_star_algorithm(main_pin_abs, sub_pin_abs)
                    if path:
                        for point in path:
                            self.wire_paths.add(point)
                        wire_paths[net_name] = path
                    else:
                        # 实在不行就用直线连接（作为最后的补救）
                        print(f"Using direct line for {net_name}")
                        wire_paths[net_name] = self._direct_line_fallback(main_pin_abs, sub_pin_abs)

        return wire_paths

    def _get_near_pin_positions(self):
        """获取pin脚附近的位置，允许布线通过"""
        near_pins = set()
        for (x, y) in self.pin_positions:
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    near_pins.add((x + dx, y + dy))
        return near_pins

    def _direct_line_fallback(self, start, end):
        """直线连接作为最后的布线方案"""
        path = []
        x0, y0 = start
        x1, y1 = end

        # 生成直线上的点
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

        # 强制添加这些点到布线，即使它们靠近器件
        for point in path:
            self.wire_paths.add(point)
            # 从占据网格中移除，允许通过
            if point in self.occupied_grid:
                self.occupied_grid.remove(point)

        return path

    def generate_pcb(self):
        """生成完整的PCB数据，确保100%布线成功率"""
        # 重置状态
        self.occupied_grid = set()
        self.wire_paths = set()
        self.pin_positions = set()
        self.pin_edges = {}

        # 1. 生成主器件
        main_comp = self.generate_main_component()

        # 2. 生成子器件
        sub_comps = self.generate_sub_components(main_comp)

        # 整合所有器件信息
        comp_info = {"main_component": main_comp}
        for sub_comp in sub_comps:
            comp_info.update(sub_comp)

        # 定义网络连接关系
        net_info = self.define_net_connections()

        # 3. 进行布线，确保所有网络都能连接
        wire_paths = self.generate_wiring(net_info, comp_info)

        # 验证所有网络都已布线
        if len(wire_paths) < len(net_info):
            missing = len(net_info) - len(wire_paths)
            # 如果有网络未布线，重试一次
            return self.generate_pcb()

        # 组织成最终的数据结构
        self.pcb_data = {
            "pcb1": {
                "component_info": comp_info,
                "net_info": net_info,
                "wire_paths": wire_paths
            }
        }

        return self.pcb_data

    def save_to_json(self, folder_path, filename=None):
        """将PCB数据保存到指定文件夹的JSON文件"""
        if not self.pcb_data:
            print("请先生成PCB数据")
            return

        # 确保文件夹存在
        os.makedirs(folder_path, exist_ok=True)

        if not filename:
            # 生成带时间戳和随机数的文件名，确保唯一性
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_suffix = random.randint(1000, 9999)
            filename = f"pcb_layout_{timestamp}_{random_suffix}.json"

        # 完整文件路径
        full_path = os.path.join(folder_path, filename)

        # 自定义编码器处理集合和元组
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, set):
                    return list(obj)
                if isinstance(obj, tuple):
                    return list(obj)
                return json.JSONEncoder.default(self, obj)

        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(self.pcb_data, f, ensure_ascii=False, indent=2, cls=CustomEncoder)

        return full_path

    def visualize(self, title="PCB布局和布线图"):
        """可视化PCB布局和布线"""
        if not self.pcb_data:
            print("请先生成PCB数据")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # 绘制PCB边界
        ax.add_patch(Rectangle((0, 0), self.pcb_size[0], self.pcb_size[1],
                               fill=False, edgecolor='black', linewidth=2))

        # 获取器件和布线信息
        comp_info = self.pcb_data["pcb1"]["component_info"]
        net_info = self.pcb_data["pcb1"]["net_info"]
        wire_paths = self.pcb_data["pcb1"]["wire_paths"]

        # 绘制主器件
        main_comp = comp_info["main_component"]
        main_x, main_y = main_comp["bottom_left_position"]
        main_width, main_height = main_comp["contour_size"]
        ax.add_patch(Rectangle((main_x, main_y), main_width, main_height,
                               fill=True, edgecolor='blue', facecolor='lightblue', alpha=0.7))
        ax.text(main_x + main_width / 2, main_y + main_height / 2, "Main",
                ha='center', va='center', fontsize=10)

        # 绘制主器件pin脚
        for pin_name, rel_pos in main_comp["pins"].items():
            pin_x = main_x + rel_pos[0]
            pin_y = main_y + rel_pos[1]
            ax.plot(pin_x, pin_y, 'bo', markersize=6)
            ax.text(pin_x + 0.3, pin_y + 0.3, pin_name, fontsize=7)

        # 绘制子器件
        sub_comp_names = [name for name in comp_info if name.startswith("sub_component")]
        colors = plt.cm.Set3(np.linspace(0, 1, len(sub_comp_names)))

        for i, sub_name in enumerate(sub_comp_names):
            sub_comp = comp_info[sub_name]
            sub_x, sub_y = sub_comp["bottom_left_position"]
            sub_width, sub_height = sub_comp["contour_size"]
            ax.add_patch(Rectangle((sub_x, sub_y), sub_width, sub_height,
                                   fill=True, edgecolor='green', facecolor=colors[i], alpha=0.7))
            ax.text(sub_x + sub_width / 2, sub_y + sub_height / 2, f"S{i + 1}",
                    ha='center', va='center', fontsize=8)

            # 绘制子器件pin脚
            for pin_name, rel_pos in sub_comp["pins"].items():
                pin_x = sub_x + rel_pos[0]
                pin_y = sub_y + rel_pos[1]
                ax.plot(pin_x, pin_y, 'go', markersize=6)
                ax.text(pin_x + 0.3, pin_y + 0.3, pin_name, fontsize=7)

        # 绘制布线
        wire_colors = plt.cm.tab10(np.linspace(0, 1, len(wire_paths)))
        for idx, (net_name, path) in enumerate(wire_paths.items()):
            if path and len(path) > 1:
                x_values = [p[0] for p in path]
                y_values = [p[1] for p in path]
                ax.plot(x_values, y_values, '-', color=wire_colors[idx], linewidth=1.5,
                        label=f"{net_name}")

        # 设置坐标轴
        ax.set_xlim(-1, self.pcb_size[0] + 1)
        ax.set_ylim(-1, self.pcb_size[1] + 1)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(title)

        # 显示网格点
        ax.set_xticks(range(self.pcb_size[0] + 1))
        ax.set_yticks(range(self.pcb_size[1] + 1))

        # 添加图例
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        return fig

    def load_from_json(self, filename):
        """从JSON文件加载PCB数据"""
        if not os.path.exists(filename):
            print(f"文件 {filename} 不存在")
            return

        with open(filename, 'r', encoding='utf-8') as f:
            self.pcb_data = json.load(f)

        # 恢复状态
        self.occupied_grid = set()
        self.wire_paths = set()
        self.pin_positions = set()

        comp_info = self.pcb_data["pcb1"]["component_info"]
        for comp_name, comp_data in comp_info.items():
            x, y = comp_data["bottom_left_position"]
            width, height = comp_data["contour_size"]

            for dx in range(width + 1):
                for dy in range(height + 1):
                    self.occupied_grid.add((x + dx, y + dy))

            for pin_rel in comp_data["pins"].values():
                pin_x = x + pin_rel[0]
                pin_y = y + pin_rel[1]
                self.pin_positions.add((pin_x, pin_y))
                if (pin_x, pin_y) in self.occupied_grid:
                    self.occupied_grid.remove((pin_x, pin_y))

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

    # 3. 创建PCB生成器实例
    pcb_gen = PCBGenerator(
        pcb_size=(30, 30),
        main_comp_min_size=(8, 8),
        main_comp_max_size=(10, 10),
        sub_comp_size=(2, 1),
        num_sub_comps=10
    )

    # 4. 生成所有数据集到临时目录
    print(f"\n开始生成 {num_datasets} 个PCB数据集（临时目录: {temp_dir}）...")

    pbar = tqdm(range(num_datasets))  # 创建进度条
    pbar.set_description("正在生成数据...")
    for i in pbar:
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
        num_datasets=100000,  # 总数据集数量（50个→训练集45个，测试集5个）
        output_folder="datasets",  # 根目录
        visualize_all=False,  # 不可视化所有
        visualize_samples=True,  # 不可视化样本（快速生成）
        sample_count=2,  # 若开启可视化，显示2个样本
        split_ratio=0.9
    )