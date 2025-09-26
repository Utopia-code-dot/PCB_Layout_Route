import os, json, torch

from torch.utils.data import Dataset


class PCBTokenizer:
    """PCB数据的token化处理类，所有特殊编码均从编码表获取"""

    def __init__(self, encode_table_path="encode_table/encode_table.json"):
        """
        初始化tokenizer

        参数:
            encode_table_path: 编码表JSON文件的路径
        """
        # 加载编码表
        self.encode_table = self._load_encode_table(encode_table_path)
        # 创建反向映射表（通过标记查找编码）
        self.decode_table = {v: k for k, v in self.encode_table.items()}

        # 定义需要的特殊标记
        self.special_token_names = [
            "<SOS>", "<EOS>", "<SOBS>", "<EOBS>",
            "<SONS>", "<EONS>", "<SODPS>", "<EODPS>",
            "<SEP>", "<SEPN>", "<SEPND>", "<PAD>"
        ]

        # 从编码表中获取特殊标记的编码
        self.special_tokens = self._get_special_tokens()

        # 获取BOARD_SIZE编码
        self.board_size_code = self._get_board_size_code()

    def _load_encode_table(self, path):
        """加载编码表JSON文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"编码表文件不存在: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            encode_table = json.load(f)

        # 将键转换为整数
        return {int(k): v for k, v in encode_table.items()}

    def _get_special_tokens(self):
        """从编码表中获取所有特殊标记的编码"""
        special_tokens = {}
        for token_name in self.special_token_names:
            if token_name not in self.decode_table:
                raise ValueError(f"编码表中缺少必要的特殊标记: {token_name}")
            special_tokens[token_name] = self.decode_table[token_name]
        return special_tokens

    def _get_position_code(self, position):
        """获取位置编码（x或y坐标）"""
        pos_str = f"POS_{position}"
        if pos_str not in self.decode_table:
            raise ValueError(f"找不到位置 {position} 的编码")
        return self.decode_table[pos_str]

    def _get_component_size_code(self, size):
        """获取器件尺寸编码（长或宽）"""
        size_str = f"COMP_SIZE_{size}"
        if size_str not in self.decode_table:
            raise ValueError(f"找不到尺寸 {size} 的编码")
        return self.decode_table[size_str]

    def _get_direction_code(self, direction):
        """获取方向编码"""
        if direction not in self.decode_table:
            raise ValueError(f"找不到方向 {direction} 的编码")
        return self.decode_table[direction]

    def _get_board_size_code(self):
        """获取PCB板尺寸编码"""
        board_size_str = "BOARD_SIZE"
        if board_size_str not in self.decode_table:
            raise ValueError("编码表中找不到BOARD_SIZE编码")
        return self.decode_table[board_size_str]

    def tokenize(self, pcb_data, max_length=None):
        """将PCB数据转换为token序列，所有编码均来自编码表"""
        # 提取PCB数据中的关键信息
        pcb_info = pcb_data["pcb1"]
        component_info = pcb_info["component_info"]
        net_info = pcb_info["net_info"]
        wire_paths = pcb_info["wire_paths"]

        main_component = component_info["main_component"]
        sub_components = {k: v for k, v in component_info.items() if k.startswith("sub_component")}

        # 获取主器件左下角坐标（用于计算pin脚绝对位置）
        main_comp_x, main_comp_y = main_component["bottom_left_position"]

        # 开始构建token序列
        tokens = []

        # 添加<SOS>标记
        tokens.append(self.special_tokens["<SOS>"])

        # 添加<SOBS>块：PCB信息
        tokens.append(self.special_tokens["<SOBS>"])

        # 添加PCB板子的轮廓大小
        tokens.append(self.board_size_code)  # PCB长度编码
        tokens.append(self.board_size_code)  # PCB宽度编码

        # 添加主器件轮廓大小
        main_comp_width, main_comp_height = main_component["contour_size"]
        tokens.append(self._get_component_size_code(main_comp_width))
        tokens.append(self._get_component_size_code(main_comp_height))

        # 添加主器件左下角的位置编码
        tokens.append(self._get_position_code(main_comp_x))
        tokens.append(self._get_position_code(main_comp_y))

        # 结束<SOBS>块
        tokens.append(self.special_tokens["<EOBS>"])

        # 添加<SEP>标记
        tokens.append(self.special_tokens["<SEP>"])

        # 添加<SONS>块：Net信息
        tokens.append(self.special_tokens["<SONS>"])

        # 按顺序添加每个net的信息
        net_names = sorted(net_info.keys(), key=lambda x: int(x.replace("net", "")))
        for i, net_name in enumerate(net_names):
            net = net_info[net_name]

            # 获取主器件pin脚信息
            main_pin_name = net["main_component"]
            main_pin_rel_x, main_pin_rel_y = main_component["pins"][main_pin_name]
            main_pin_abs_x = main_comp_x + main_pin_rel_x
            main_pin_abs_y = main_comp_y + main_pin_rel_y
            tokens.append(self._get_position_code(main_pin_abs_x))
            tokens.append(self._get_position_code(main_pin_abs_y))

            # 获取子器件信息
            sub_comp_name = [k for k in net.keys() if k.startswith("sub_component")][0]
            sub_comp = sub_components[sub_comp_name]

            # 获取子器件轮廓大小
            sub_comp_width, sub_comp_height = sub_comp["contour_size"]
            tokens.append(self._get_component_size_code(sub_comp_width))
            tokens.append(self._get_component_size_code(sub_comp_height))

            # 添加<SEPN>标记
            if i < len(net_names) - 1:
                tokens.append(self.special_tokens["<SEPN>"])

        # 结束<SONS>块
        tokens.append(self.special_tokens["<EONS>"])

        # 添加<SEP>标记
        tokens.append(self.special_tokens["<SEP>"])

        # 添加<SODPS>块：布线信息
        tokens.append(self.special_tokens["<SODPS>"])

        # 按顺序添加每个net的布线路径信息
        for i, net_name in enumerate(net_names):
            # 获取布线路径信息
            wire_path = wire_paths.get(net_name, [])

            # 提取路径中的方向信息
            if len(wire_path) >= 2:
                directions = []
                for j in range(1, len(wire_path)):
                    dx = wire_path[j][0] - wire_path[j - 1][0]
                    dy = wire_path[j][1] - wire_path[j - 1][1]

                    if dx > 0 and dy == 0:
                        directions.append("DIR_RIGHT")
                    elif dx < 0 and dy == 0:
                        directions.append("DIR_LEFT")
                    elif dx == 0 and dy > 0:
                        directions.append("DIR_UP")
                    elif dx == 0 and dy < 0:
                        directions.append("DIR_DOWN")
                    elif dx > 0 and dy > 0:
                        directions.append("DIR_UP_RIGHT")
                    elif dx < 0 and dy > 0:
                        directions.append("DIR_UP_LEFT")
                    elif dx > 0 and dy < 0:
                        directions.append("DIR_DOWN_RIGHT")
                    elif dx < 0 and dy < 0:
                        directions.append("DIR_DOWN_LEFT")

                # 将方向转换为编码并添加
                for direction in directions[:10]:
                    tokens.append(self._get_direction_code(direction))

            # 获取子器件左下角位置编码
            sub_comp_name = [k for k in net_info[net_name].keys() if k.startswith("sub_component")][0]
            sub_comp = sub_components[sub_comp_name]
            sub_comp_x, sub_comp_y = sub_comp["bottom_left_position"]
            tokens.append(self._get_position_code(sub_comp_x))
            tokens.append(self._get_position_code(sub_comp_y))

            # 添加<SEPDP>标记
            if i < len(net_names) - 1:
                tokens.append(self.special_tokens["<SEPND>"])

        # 结束<SODPS>块
        tokens.append(self.special_tokens["<EODPS>"])

        # 添加<EOS>标记
        tokens.append(self.special_tokens["<EOS>"])

        # 处理最大长度和填充
        if max_length is not None:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
                if tokens[-1] != self.special_tokens["<EOS>"]:
                    tokens[-1] = self.special_tokens["<EOS>"]
            else:
                tokens += [self.special_tokens["<PAD>"]] * (max_length - len(tokens))

        return tokens


class PCBDataset(Dataset):
    """PCB数据集类，用于加载和处理PCB数据，兼容PyTorch的DataLoader"""

    def __init__(self, data_dir, tokenizer, max_length=1024):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 获取所有数据文件的路径（仅JSON格式）
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

        if not self.data_files:
            raise ValueError(f"在目录 {data_dir} 中未找到任何JSON数据文件（请确认预划分的数据集已生成）")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data_files):
            raise IndexError("索引超出范围")

        # 加载PCB数据
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        with open(file_path, 'r', encoding='utf-8') as f:
            pcb_data = json.load(f)

        # 进行token化
        tokens = self.tokenizer.tokenize(pcb_data, self.max_length)

        # 转换为PyTorch张量
        token_tensor = torch.tensor(tokens, dtype=torch.long)

        return {
            "tokens": token_tensor,
            "filename": self.data_files[idx]
        }