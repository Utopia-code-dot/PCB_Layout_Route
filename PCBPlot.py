import torch, matplotlib
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from PCBTokenizer import PCBTokenizer
from config import config

class PCBPlot(object):
    def __init__(self, tokens):
        self.tokens = tokens[0].tolist()
        self.tokenizer = PCBTokenizer()

        self.get_layout_info()

    def get_layout_info(self):
        self.layout_info = {"pcb": list(), "comp":{"main_comp":{"contour": [], "pos": [], "pin":{}}, "sub_comp":{}}, "net": {}, "wire_path": {}}

        n_sub_contour, n_sub_pos = 1, 1
        n_net = 1
        n_main_pin, n_sub_pin = 1, 1
        n_line = 1
        sons, sodps = False, False
        pos_flag, contour_flag = False, False
        for i in range(len(self.tokens)):
            token = self.tokens[i]

            # 提取pcb大小
            if self.tokenizer.encode_table[token] == "BOARD_SIZE":
                self.layout_info["pcb"].append(int(token - len(self.tokenizer.special_tokens)))
            # 提取器件轮廓
            elif self.tokenizer.encode_table[token].split("_")[0] == "COMP":
                # 提取主器件轮廓
                if not sons and not sodps:
                    if len(self.layout_info["comp"]["main_comp"]["contour"]) < 2:
                        self.layout_info["comp"]["main_comp"]["contour"].append(int(self.tokenizer.encode_table[token].split("_")[-1]))
                # 提取子器件轮廓
                elif sons and not sodps:
                    # 查看子器件信息是否被创建
                    if not contour_flag:
                        # 没创建则创建子器件信息
                        if f"sub_comp_{n_sub_contour}" not in list(self.layout_info["comp"]["sub_comp"].keys()):
                            self.layout_info["comp"]["sub_comp"][f"sub_comp_{n_sub_contour}"] = {"contour": [], "pos": [], "pin": {}}
                        contour_flag = True

                    self.layout_info["comp"]["sub_comp"][f"sub_comp_{n_sub_contour}"]["contour"].append(int(self.tokenizer.encode_table[token].split("_")[-1]))

                    # 查看contour信息是否完整
                    if len(self.layout_info["comp"]["sub_comp"][f"sub_comp_{n_sub_contour}"]["contour"]) >= 2:
                        contour_flag = False
                        n_sub_contour += 1  # contour信息完整
                else:
                    print("器件轮廓编码的位置出错")
            # 提取位置
            elif self.tokenizer.encode_table[token].split("_")[0] == "POS":
                # 提取主器件位置
                if not sons and not sodps:
                    if len(self.layout_info["comp"]["main_comp"]["pos"]) < 2:
                        self.layout_info["comp"]["main_comp"]["pos"].append(int(self.tokenizer.encode_table[token].split("_")[-1]))
                # 提取主器件pin脚位置
                elif sons and not sodps:
                    # 查看pin脚信息是否被创建
                    if not pos_flag:
                        # 没创建则创建pin脚信息
                        self.layout_info["comp"]["main_comp"]["pin"][f"pin_{n_main_pin}"] = list()
                        pos_flag = True
                        # 创建net信息
                        self.layout_info["net"][f"net_{n_net}"] = [f"pin_{n_main_pin}", "pin_1"]
                        self.layout_info["wire_path"][f"net_{n_net}"] = list()
                        n_net += 1

                    self.layout_info["comp"]["main_comp"]["pin"][f"pin_{n_main_pin}"].append(int(self.tokenizer.encode_table[token].split("_")[-1]))

                    # 查看pos信息是否完整
                    if len(self.layout_info["comp"]["main_comp"]["pin"][f"pin_{n_main_pin}"]) >= 2:
                        pos_flag = False
                        n_main_pin += 1  # contour信息完整
                # 后续的为子器件位置
                else:
                    # 查看子器件信息是否被创建
                    if not pos_flag:
                        # 没创建则创建子器件信息
                        if f"sub_comp_{n_sub_pos}" not in list(self.layout_info["comp"]["sub_comp"].keys()):
                            self.layout_info["comp"]["sub_comp"][f"sub_comp_{n_sub_pos}"] = {"contour": [], "pos": [], "pin": {}}
                        pos_flag = True

                    self.layout_info["comp"]["sub_comp"][f"sub_comp_{n_sub_pos}"]["pos"].append(int(self.tokenizer.encode_table[token].split("_")[-1]))

                    # 查看pos信息是否完整
                    if len(self.layout_info["comp"]["sub_comp"][f"sub_comp_{n_sub_pos}"]["pos"]) >= 2:
                        pos_flag = False
                        n_sub_pos += 1  # contour信息完整
            # 提取布线信息
            elif self.tokenizer.encode_table[token].split("_")[0] == "DIR":
                if self.tokenizer.encode_table[self.tokens[i+1]].split("_")[0] != "DIR":
                    n_line += 1
                else:
                    self.layout_info["wire_path"][f"net_{n_line}"].append(token)
            # 特殊编码
            elif self.tokenizer.encode_table[token] == "<SONS>":
                sons = True
            elif self.tokenizer.encode_table[token] == "<SODPS>":
                sodps = True

    def plot_token(self):
        """
        将token序列绘制为布局结果
        :return:
        """
        # fig, axes = plt.subplots(figsize(10, 10))
        plt.figure(figsize=(10, 10))
        # 绘制主器件
        main_pos_x, main_pos_y = self.layout_info["comp"]["main_comp"]["pos"]  # 左下角坐标
        main_contour_x, main_contour_y = self.layout_info["comp"]["main_comp"]["contour"]  # 轮廓
        # 绘制轮廓
        x1, x2 = main_pos_x, main_pos_x + main_contour_x
        y1, y2 = main_pos_y, main_pos_y + main_contour_y
        plt.plot([x1, x2], [y1, y1], color=config["main_comp_color"], linewidth=config["comp_line_width"])
        plt.plot([x2, x2], [y1, y2], color=config["main_comp_color"], linewidth=config["comp_line_width"])
        plt.plot([x1, x2], [y2, y2], color=config["main_comp_color"], linewidth=config["comp_line_width"])
        plt.plot([x1, x1], [y1, y2], color=config["main_comp_color"], linewidth=config["comp_line_width"])
        # # 绘制pin脚
        # for pin, pos in self.layout_info["comp"]["main_comp"]["pin"].items():
        #     main_pin = plt.Circle((pos[0], pos[1]), config["pin_size"], color=config["pin_color"])
        #     axes.set_aspect(1)
        #     axes.add_artist(main_pin)

        # 绘制子器件
        for comp_name, comp_info in self.layout_info["comp"]["sub_comp"].items():
            sub_pos_x, sub_pos_y = comp_info["pos"]
            sub_contour_x, sub_contour_y = comp_info["contour"]
            # 绘制轮廓
            x1, x2 = sub_pos_x, sub_pos_x + sub_contour_x
            y1, y2 = sub_pos_y, sub_pos_y + sub_contour_y
            plt.plot([x1, x2], [y1, y1], color=config["sub_comp_color"], linewidth=config["comp_line_width"])
            plt.plot([x2, x2], [y1, y2], color=config["sub_comp_color"], linewidth=config["comp_line_width"])
            plt.plot([x1, x2], [y2, y2], color=config["sub_comp_color"], linewidth=config["comp_line_width"])
            plt.plot([x1, x1], [y1, y2], color=config["sub_comp_color"], linewidth=config["comp_line_width"])

        plt.xlim((0, self.layout_info["pcb"][0]))
        plt.ylim((0, self.layout_info["pcb"][1]))
        plt.show()



tokens = torch.tensor([[ 1,  3, 42, 42, 55, 50, 21, 24,  4,  9,  5, 25, 25, 43, 45, 10, 29, 25,
        46, 46, 10, 33, 26, 43, 43, 10, 33, 29, 43, 45, 10, 25, 32, 46, 44, 10,
        29, 32, 43, 46, 10, 21, 26, 46, 44, 10, 21, 29, 46, 46,  6,  9,  7, 78,
        73, 73, 73, 73, 24, 17, 11, 78, 78, 73, 73, 27, 17, 11, 75, 75, 75, 75,
        37, 25, 11, 75, 75, 75, 75, 37, 28, 11, 77, 72, 72, 72, 24, 35, 11, 77,
        72, 72, 72, 30, 35, 11, 78, 74, 74, 74, 14, 24, 11, 76, 76, 76, 74, 14,
        28,  8,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0]])

a = PCBPlot(tokens)
a.plot_token()

