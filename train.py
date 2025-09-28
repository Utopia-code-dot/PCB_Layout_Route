import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import config
from model import DecoderOnlyTransformer
from PCBTokenizer import PCBTokenizer, PCBDataset


# 训练函数（包含Token正确率统计）
def train_model(model, train_loader, criterion, optimizer, device, pad_token):
    """训练一个epoch，返回平均损失和Token正确率"""
    model.train()
    total_loss = 0.0
    total_correct = 0  # 统计正确的Token数
    total_valid_tokens = 0  # 统计有效Token数（排除PAD）

    for batch in tqdm(train_loader, desc="Training"):
        tokens = batch["tokens"].to(device)

        # 前n-1个token作为输入，后n-1个token作为目标
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]

        # 清零梯度
        optimizer.zero_grad()

        # 生成目标掩码（防止模型看到未来token）
        tgt_len = input_tokens.size(1)
        tgt_mask = model._generate_square_subsequent_mask(tgt_len).to(device)

        # 前向传播
        output = model(input_tokens, tgt_mask=tgt_mask)  # (batch_size, seq_len-1, vocab_size)

        # 1. 计算损失（忽略<PAD>标记）
        loss = criterion(output.transpose(1, 2), target_tokens)  # (batch_size, seq_len-1)
        mask = (target_tokens != pad_token).float()  # (batch_size, seq_len-1)：PAD位置为0，有效位置为1
        loss = (loss * mask).sum() / mask.sum()  # 加权平均损失

        # 2. 计算Token正确率（仅统计有效Token）
        pred_tokens = torch.argmax(output, dim=-1)  # (batch_size, seq_len-1)：取概率最大的Token
        correct = (pred_tokens == target_tokens) & (target_tokens != pad_token)  # 正确且非PAD的位置
        total_correct += correct.sum().item()  # 累加正确数
        total_valid_tokens += mask.sum().item()  # 累加有效Token数

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 计算平均损失和正确率
    avg_loss = total_loss / len(train_loader)
    accuracy = (total_correct / total_valid_tokens) * 100 if total_valid_tokens > 0 else 0.0
    return avg_loss, accuracy


# 验证函数（包含Token正确率统计）
def validate_model(model, val_loader, criterion, device, pad_token):
    """验证模型，返回平均损失和Token正确率"""
    model.eval()
    total_loss = 0.0
    total_correct = 0  # 统计正确的Token数
    total_valid_tokens = 0  # 统计有效Token数（排除PAD）

    with torch.no_grad():  # 禁用梯度计算，加快验证速度并避免内存占用
        for batch in tqdm(val_loader, desc="Validation"):
            tokens = batch["tokens"].to(device)

            # 前n-1个token作为输入，后n-1个token作为目标（与训练逻辑一致）
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            # 生成目标掩码
            tgt_len = input_tokens.size(1)
            tgt_mask = model._generate_square_subsequent_mask(tgt_len).to(device)

            # 前向传播
            output = model(input_tokens, tgt_mask=tgt_mask)  # (batch_size, seq_len-1, vocab_size)

            # 1. 计算损失（忽略<PAD>）
            loss = criterion(output.transpose(1, 2), target_tokens)
            mask = (target_tokens != pad_token).float()
            loss = (loss * mask).sum() / mask.sum()

            # 2. 计算Token正确率（仅统计有效Token）
            pred_tokens = torch.argmax(output, dim=-1)
            correct = (pred_tokens == target_tokens) & (target_tokens != pad_token)
            total_correct += correct.sum().item()
            total_valid_tokens += mask.sum().item()

            total_loss += loss.item()

    # 计算平均损失和正确率
    avg_loss = total_loss / len(val_loader)
    accuracy = (total_correct / total_valid_tokens) * 100 if total_valid_tokens > 0 else 0.0
    return avg_loss, accuracy

# 测试函数
def predict_route(model, tokenizer, prompt_tokens, max_length, device):
    """
    预测PCB布线信息（从<SOS>到<SODPS>为提示，生成后续布线token）
    """
    model.eval()

    # 将提示移到设备上
    prompt = prompt_tokens.to(device)

    # 生成预测（使用预划分的编码表特殊标记）
    generated = model.generate(
        prompt,
        max_length=max_length,
        pad_token=tokenizer.special_tokens["<PAD>"],
        eos_token=tokenizer.special_tokens["<EOS>"]
    )

    return generated.cpu()  # 转回CPU以便后续处理


def test_model(model, test_loader, criterion, device, pad_token):
    """测试模型，返回平均损失和Token正确率"""
    model.eval()
    total_loss = 0.0
    total_correct = 0  # 统计正确的Token数
    total_valid_tokens = 0  # 统计有效Token数（排除PAD）

    with torch.no_grad():  # 禁用梯度计算
        for batch in tqdm(test_loader, desc="Testing"):
            tokens = batch["tokens"].to(device)

            # 前n-1个token作为输入，后n-1个token作为目标
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            # 生成目标掩码
            tgt_len = input_tokens.size(1)
            tgt_mask = model._generate_square_subsequent_mask(tgt_len).to(device)

            # 前向传播
            output = model(input_tokens, tgt_mask=tgt_mask)

            # 计算损失（忽略<PAD>）
            loss = criterion(output.transpose(1, 2), target_tokens)
            mask = (target_tokens != pad_token).float()
            loss = (loss * mask).sum() / mask.sum()

            # 计算Token正确率
            pred_tokens = torch.argmax(output, dim=-1)
            correct = (pred_tokens == target_tokens) & (target_tokens != pad_token)
            total_correct += correct.sum().item()
            total_valid_tokens += mask.sum().item()

            total_loss += loss.item()

    # 计算平均损失和正确率
    avg_loss = total_loss / len(test_loader)
    accuracy = (total_correct / total_valid_tokens) * 100 if total_valid_tokens > 0 else 0.0
    return avg_loss, accuracy


def find_token_position(tokens, token_value):
    """找到特定token在序列中的第一个出现位置（用于提取预测提示）"""
    positions = (tokens == token_value).nonzero()
    if positions.numel() == 0:
        return -1  # 未找到目标token
    return positions[0, 1].item()  # 返回第一个匹配位置（按batch第一个样本）


def train(train_flag=True):
    if not train_flag:
        return

    # 创建必要的文件夹（如果不存在）
    os.makedirs("pic/train", exist_ok=True)
    os.makedirs("ckpt", exist_ok=True)

    # --------------------------------------------------------------------------------
    print("=" * 60)
    print("PCB Transformer 训练配置")
    print("=" * 60)
    print(f"训练集目录: {os.path.abspath(config['train_dir'])}")
    print(f"测试集目录: {os.path.abspath(config['test_dir'])}")
    print(f"使用设备: {config['device']}")
    print(f"序列最大长度: {config['max_length']} | 批次大小: {config['batch_size']}")

    # -------------------------- 1. 初始化Tokenizer --------------------------
    try:
        tokenizer = PCBTokenizer(encode_table_path=config["encode_table_path"])
        print(f"\n成功加载编码表，词汇表大小: {len(tokenizer.encode_table)}")
    except Exception as e:
        print(f"\n初始化Tokenizer失败: {str(e)}")
        return

    # -------------------------- 2. 加载训练数据并划分训练集和验证集 --------------------------
    try:
        # 加载完整训练数据
        full_train_dataset = PCBDataset(
            data_dir=config["train_dir"],
            tokenizer=tokenizer,
            max_length=config["max_length"]
        )

        # 划分训练集和验证集
        val_size = int(0.1 * len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # 固定随机种子，确保划分一致
        )

        print(f"\n成功加载并划分训练数据:")
        print(f"  总训练样本数: {len(full_train_dataset)} | 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")
    except Exception as e:
        print(f"\n加载训练数据失败: {str(e)}")
        return

    # -------------------------- 3. 创建DataLoader --------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,  # 训练集打乱以保证泛化性
        num_workers=4,
        pin_memory=True
    )

    # 创建验证集DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,  # 验证集不打乱
        num_workers=4,
        pin_memory=True
    )

    # -------------------------- 4. 创建Transformer模型 --------------------------
    try:
        vocab_size = len(tokenizer.encode_table)
        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"]
        ).to(config["device"])

        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTransformer模型创建完成（{config['device']}）")
        print(f"  模型总参数数: {total_params / 1e6:.2f}M")
    except Exception as e:
        print(f"\n创建模型失败: {str(e)}")
        return

    # -------------------------- 5. 定义损失函数与优化器 --------------------------
    criterion = nn.CrossEntropyLoss(reduction='none')  # 不自动归约，后续手动处理<PAD>
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    # 学习率调度器：验证损失3轮不下降则学习率减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=3,
        factor=0.5,
        verbose=True
    )

    # -------------------------- 6. 训练模型（每10轮验证一次） --------------------------
    print(f"\n开始训练")
    best_val_loss = float('inf')  # 记录最佳验证损失
    # 记录每轮的损失和正确率
    train_losses = []
    train_accs = []  # 训练集Token正确率
    val_losses = []
    val_accs = []  # 验证集Token正确率
    val_epochs = []  # 记录进行验证的轮次

    # 创建验证数据迭代器
    val_iter = iter(val_loader)

    for epoch in range(config["epochs"]):
        print(f"\n=== Epoch {epoch + 1}/{config['epochs']} ===")

        # 训练一轮（返回损失和正确率）
        train_loss, train_acc = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=config["device"],
            pad_token=tokenizer.special_tokens["<PAD>"]
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"训练损失: {train_loss:.4f} | 训练Token正确率: {train_acc:.2f}%")

        # 每10轮进行一次验证（或最后一轮）
        if (epoch + 1) % 10 == 0 or (epoch + 1) == config["epochs"]:
            # 尝试获取下一个验证批次，如果迭代器耗尽则重新创建
            try:
                val_batch = next(val_iter)
                # 创建临时验证加载器
                temp_val_loader = DataLoader(
                    val_dataset,
                    batch_size=len(val_batch),
                    shuffle=False
                )
                val_loss, val_acc = validate_model(
                    model=model,
                    val_loader=temp_val_loader,
                    criterion=criterion,
                    device=config["device"],
                    pad_token=tokenizer.special_tokens["<PAD>"]
                )
            except StopIteration:
                # 迭代器耗尽，重新创建
                val_iter = iter(val_loader)
                val_batch = next(val_iter)
                temp_val_loader = DataLoader(
                    val_dataset,
                    batch_size=len(val_batch),
                    shuffle=False
                )
                val_loss, val_acc = validate_model(
                    model=model,
                    val_loader=temp_val_loader,
                    criterion=criterion,
                    device=config["device"],
                    pad_token=tokenizer.special_tokens["<PAD>"]
                )

            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_epochs.append(epoch + 1)
            print(f"验证损失: {val_loss:.4f} | 验证Token正确率: {val_acc:.2f}%")

            # 调整学习率（基于验证损失）
            scheduler.step(val_loss)

            # 保存最佳模型（仅当验证损失下降时保存）
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_acc': val_acc,
                    'config': config
                }, config["save_path"])
                print(
                    f"保存最佳模型（验证损失: {best_val_loss:.4f} | 验证正确率: {val_acc:.2f}%）到 {config['save_path']}")

        # 绘制并保存当前的损失和正确率曲线
        try:
            plt.figure(figsize=(12, 6))
            # 绘制损失曲线
            plt.plot(range(1, len(train_losses) + 1), train_losses, label='train loss', linewidth=2)
            plt.plot(val_epochs, val_losses, label='val loss', linewidth=2)

            # 绘制正确率曲线（双Y轴）
            ax2 = plt.gca().twinx()
            ax2.plot(range(1, len(train_accs) + 1), train_accs, 'r--', label='train token acc', linewidth=2)
            ax2.plot(val_epochs, val_accs, 'orange', label='val token acc', linewidth=2)

            # 坐标轴设置
            plt.gca().set_xlabel('Epoch', fontsize=12)
            plt.gca().set_ylabel('loss', fontsize=12, color='black')
            ax2.set_ylabel('Token acc（%）', fontsize=12, color='red')
            plt.gca().set_title(f'PCB Transformer loss&token loss（Epoch {epoch + 1}）', fontsize=14,
                                fontweight='bold')

            # 合并图例
            lines1, labels1 = plt.gca().get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            plt.gca().legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')

            plt.grid(True, alpha=0.3)
            plt.savefig(f'pic/train/loss_acc_curve_epoch_{epoch + 1}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"绘制曲线失败: {str(e)}")

    # -------------------------- 训练完成 --------------------------
    print(f"\n训练流程全部完成！")
    print(f"关键输出:")
    print(f"  - 最佳模型: {os.path.abspath(config['save_path'])}")
    print(f"  - 最终训练正确率: {train_accs[-1]:.2f}%")
    if val_accs:
        print(f"  - 最终验证正确率: {val_accs[-1]:.2f}%")


def test(test_flag=True):
    if not test_flag:
        return

    # -------------------------- 加载最佳模型进行测试 --------------------------
    print("=" * 60)
    print("PCB Transformer 测试配置")
    print("=" * 60)
    print(f"测试集目录: {os.path.abspath(config['test_dir'])}")
    print(f"使用设备: {config['device']}")

    # -------------------------- 1. 初始化Tokenizer --------------------------
    try:
        tokenizer = PCBTokenizer(encode_table_path=config["encode_table_path"])
        print(f"\n成功加载编码表，词汇表大小: {len(tokenizer.encode_table)}")
    except Exception as e:
        print(f"\n初始化Tokenizer失败: {str(e)}")
        return

    # -------------------------- 2. 加载测试数据 --------------------------
    try:
        # 加载测试集
        test_dataset = PCBDataset(
            data_dir=config["test_dir"],
            tokenizer=tokenizer,
            max_length=config["max_length"]
        )

        print(f"\n成功加载测试数据集:")
        print(f"  测试集样本数: {len(test_dataset)}")
    except Exception as e:
        print(f"\n加载测试数据集失败: {str(e)}")
        return

    # -------------------------- 3. 创建测试DataLoader --------------------------
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,  # 测试集不打乱
        num_workers=4,
        pin_memory=True
    )

    # -------------------------- 4. 加载模型 --------------------------
    try:
        vocab_size = len(tokenizer.encode_table)
        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"]
        ).to(config["device"])

        # 加载最佳模型权重
        checkpoint = torch.load(config["save_path"], map_location=config["device"])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n成功加载最佳模型（训练轮数: {checkpoint['epoch']}）")
    except Exception as e:
        print(f"\n加载模型失败: {str(e)}")
        return

    # -------------------------- 5. 进行测试 --------------------------
    total_token_acc = 0

    try:
        for batch in tqdm(test_loader, desc="Testing"):
            sample_tokens = batch["tokens"].to(config["device"])
            sample_filename = batch["filename"]
            print(f"\n预测示例（测试集样本: {sample_filename}）")

            # 提取提示序列：从<SOS>到<SODPS>
            sodps_token = tokenizer.special_tokens["<SODPS>"]
            sodps_pos = find_token_position(sample_tokens, sodps_token)
            if sodps_pos == -1:
                print(f"样本 {sample_filename} 中未找到<SODPS>标记，无法生成提示")
                return

            # 提示序列：包含<SOS>到<SODPS>的所有token
            prompt_tokens = sample_tokens[:, :sodps_pos + 1].to(config["device"])

            # 计算最大生成长度
            max_gen_length = config["max_length"] - prompt_tokens.size(1)
            if max_gen_length <= 0:
                print(f"提示序列长度已达到max_length，无法生成更多token")
                return

            # 生成布线信息
            generated_tokens = predict_route(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_length=max_gen_length,
                device=config["device"]
            )

            # 打印预测结果
            print(f"\n预测结果:")
            print(f"  标签token：{sample_tokens[0, sodps_pos + 1:]}")
            print(f"  生成token: {generated_tokens[0, sodps_pos + 1:]}")


            # 计算Token正确率
            token_acc = sum((sample_tokens[0, sodps_pos + 1:] == generated_tokens[0, sodps_pos + 1:]) & (sample_tokens[0, sodps_pos + 1:] != 0).sum()) / torch.count_nonzero(sample_tokens[0, sodps_pos + 1:])
            total_token_acc += token_acc / len(test_dataset)
            print(f"  token准确率：{token_acc}")

        print(f"token正确率：{total_token_acc}")

    except Exception as e:
        print(f"\n预测示例失败: {str(e)}")
        return

    # -------------------------- 测试完成 --------------------------
    print(f"\n测试流程全部完成！")


if __name__ == "__main__":
    # 可以通过修改这两个标志来控制执行训练还是测试
    train_flag = False
    test_flag = True

    if train_flag:
        train()
    if test_flag:
        test()
