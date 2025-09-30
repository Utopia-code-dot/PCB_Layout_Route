import  torch


# -------------------------- 配置参数 --------------------------
config = {
    "train_dir": "datasets/train_data",  # 预划分的训练集目录
    "test_dir": "datasets/test_data",  # 预划分的验证集目录
    "encode_table_path": "encode_table/encode_table.json",  # 编码表路径
    "max_length": 256,  # 序列最大长度
    "batch_size": 8,  # 训练批次大小
    "d_model": 256,  # Transformer模型维度
    "nhead": 8,  # 多头注意力头数
    "num_layers": 6,  # 解码器层数
    "dim_feedforward": 256,  # 前馈网络隐藏层维度
    "dropout": 0.1,  # Dropout概率
    "epochs": 150,  # 训练轮数
    "lr": 1e-4,  # 初始学习率
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # 自动选择设备
    "save_path": "ckpt/pcb_transformer_model.pth",  # 模型保存路径（调整到ckpt文件夹）

    "main_comp_color": "red",
    "sub_comp_color": "blue",
    "pin_color": "black",
    "comp_line_width": 2,
    "wire_width": 2,
    "pin_size": 0.5,
}