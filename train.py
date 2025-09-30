import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from config import config
from model import GPTModel
from PCBTokenizer import PCBTokenizer, PCBDataset


# 训练函数（包含Token正确率统计）
def train_model(model, train_loader, criterion, optimizer, scheduler, device, pad_token):
    """训练一个epoch，返回平均损失和Token正确率"""
    model.train()
    total_loss = 0.0
    total_correct = 0  # 统计正确的Token数
    total_tokens = 0  # 统计有效Token数（排除PAD）

    for batch in tqdm(train_loader, desc="Training"):
        tokens = batch["tokens"].to(device)
        print(tokens[0])
        # 前n-1个token作为输入，后n-1个token作为目标
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]

        # 清零梯度
        optimizer.zero_grad()

        # 通过模型计算输出
        logits = model(input_tokens)
        logits = logits.reshape(-1, logits.size(-1))
        preds = logits.argmax(dim=-1)

        # 处理标签便于计算损失
        target_tokens = target_tokens.reshape(-1)
        seq_length = tokens.size(1) - 1

        # 掩码机制
        cond_len = find_cond_length(tokens[0, :])
        mask_indices = torch.arange(seq_length, device=device) >= (cond_len - 1)
        mask_indices = mask_indices.unsqueeze(0).expand(tokens.size(0), -1).reshape(-1)
        loss_mask = mask_indices.bool() & (target_tokens != pad_token)
        logits_masked = logits[loss_mask]
        target_masked = target_tokens[loss_mask]

        # 计算损失
        loss = criterion(logits_masked, target_masked)
        loss.backward()

        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 梯度更新
        optimizer.step()

        # 计算正确率
        with torch.no_grad():
            total_loss += loss.item()
            acc_mask = (target_tokens != pad_token) & mask_indices
            total_correct += (preds[acc_mask] == target_tokens[acc_mask]).float().sum().item()
            total_tokens += acc_mask.sum().item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    epoch_acc = total_correct / total_tokens * 100 if total_tokens > 0 else 0.0  # 转换为百分比
    return avg_loss, epoch_acc


# 验证函数（包含Token正确率统计）
def validate_model(model, val_loader, criterion, device, pad_token):
    """验证模型，返回平均损失和Token正确率"""
    model.eval()
    total_loss = 0.0
    total_correct = 0  # 统计正确的Token数
    total_tokens = 0  # 统计有效Token数（排除PAD）

    with torch.no_grad():  # 禁用梯度计算，加快验证速度并避免内存占用
        for batch in tqdm(val_loader, desc="Validation"):
            tokens = batch["tokens"].to(device)

            # 前n-1个token作为输入，后n-1个token作为目标（与训练逻辑一致）
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            # 通过模型计算输出
            logits = model(input_tokens)
            logits = logits.reshape(-1, logits.size(-1))
            preds = logits.argmax(dim=-1)

            # 处理标签便于计算损失
            target_tokens = target_tokens.reshape(-1)
            seq_length = tokens.size(1) - 1

            # 掩码机制
            cond_len = find_cond_length(tokens[0, :])
            mask_indices = torch.arange(seq_length, device=device) >= (cond_len - 1)
            mask_indices = mask_indices.unsqueeze(0).expand(tokens.size(0), -1).reshape(-1)
            loss_mask = mask_indices.bool() & (target_tokens != pad_token)
            logits_masked = logits[loss_mask]
            target_masked = target_tokens[loss_mask]

            # 计算损失
            loss = criterion(logits_masked, target_masked)

            total_loss += loss.item()
            acc_mask = (target_tokens != pad_token) & mask_indices
            total_correct += (preds[acc_mask] == target_tokens[acc_mask]).float().sum().item()
            total_tokens += acc_mask.sum().item()

    # 计算平均损失和正确率
    avg_loss = total_loss / len(val_loader)
    epoch_acc = total_correct / total_tokens * 100 if total_tokens > 0 else 0.0  # 转换为百分比
    return avg_loss, epoch_acc


def test_model(model, prompt_tokens, sample_tokens, device, max_gen_len, mask=False):
    """测试模型，返回平均损失和Token正确率"""
    model.eval()
    output = prompt_tokens.copy()

    with torch.no_grad():  # 禁用梯度计算
        for i in range(max_gen_len):
            # 预测下一个token
            inp = output
            logits = model(inp)
            next_token = int(torch.argmax(logits[0, -1, :]).item())

            if mask:
                if len(prompt_tokens) + i < len(sample_tokens):
                    label_token = sample_tokens[len(prompt_tokens) + i]  # 该位置的label
                    if label_token == 11:
                        next_token = label_token

            output = torch.cat((output, torch.tensor([[next_token]]).to(device)), dim=1)

    return output


def find_cond_length(tokens):
    cnt = 0
    for token in tokens:
        if token == 7:
            return cnt + 1
        else:
            cnt += 1
    return -1


def find_token_position(tokens, token_value):
    """找到特定token在序列中的第一个出现位置（用于提取预测提示）"""
    try:
        return tokens.tolist().index(token_value)  # 返回第一个匹配位置（按batch第一个样本）
    except:
        return -1  # 未找到目标token


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

    # -------------------------- 初始化Tokenizer --------------------------
    try:
        tokenizer = PCBTokenizer(encode_table_path=config["encode_table_path"])
        print(f"\n成功加载编码表，词汇表大小: {len(tokenizer.encode_table)}")
    except Exception as e:
        print(f"\n初始化Tokenizer失败: {str(e)}")
        return

    # -------------------------- 加载训练数据并划分训练集和验证集 --------------------------
    try:
        # 加载完整训练数据
        full_train_dataset = PCBDataset(data_dir=config["train_dir"], tokenizer=tokenizer, max_length=config["max_length"])

        # 划分训练集和验证集
        val_size = int(0.1 * len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))  # 固定随机种子，确保划分一致

        print(f"\n成功加载并划分训练数据:")
        print(f"  总训练样本数: {len(full_train_dataset)} | 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")
    except Exception as e:
        print(f"\n加载训练数据失败: {str(e)}")
        return

    # -------------------------- 创建DataLoader --------------------------
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # -------------------------- 创建Transformer模型 --------------------------
    try:
        vocab_size = len(tokenizer.encode_table)
        model = GPTModel(vocab_size, config["d_model"], config["num_layers"], config["nhead"], config["dim_feedforward"], config["dropout"]).to(config["device"])

        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTransformer模型创建完成（{config['device']}）")
        print(f"  模型总参数数: {total_params / 1e6:.2f}M")
    except Exception as e:
        print(f"\n创建模型失败: {str(e)}")
        return

    # -------------------------- 定义损失函数与优化器 --------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # -------------------------- 训练模型（每10轮验证一次） --------------------------
    print(f"\n开始训练")
    best_val_loss = float('inf')  # 记录最佳验证损失

    # 记录每轮的损失和正确率
    train_losses = []
    train_accs = []  # 训练集Token正确率
    val_losses = []
    val_accs = []  # 验证集Token正确率
    val_epochs = []  # 记录进行验证的轮次

    for epoch in range(config["epochs"]):
        print(f"\n=== Epoch {epoch + 1}/{config['epochs']} ===")

        # 训练一轮（返回损失和正确率）
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, scheduler, config["device"], tokenizer.special_tokens["<PAD>"])
        print(f"训练损失: {train_loss:.4f} | 训练Token正确率: {train_acc:.2f}%")

        # 记录训练损失和正确率
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 每10轮进行一次验证（或最后一轮）
        if (epoch + 1) % 10 == 0 or (epoch + 1) == config["epochs"]:
            val_loss, val_acc = validate_model(model, val_loader, criterion, config["device"], tokenizer.special_tokens["<PAD>"])
            print(f"验证损失: {val_loss:.4f} | 验证Token正确率: {val_acc:.2f}%")

            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_epochs.append(epoch + 1)

            # 调整学习率
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

                # 创建第一个Y轴（左侧）用于损失
                ax1 = plt.gca()
                # 绘制训练损失和验证损失
                ax1.plot(range(1, len(train_losses) + 1), train_losses, 'blue', label='train loss', linewidth=2)
                ax1.plot(val_epochs, val_losses, 'green', label='val loss', linewidth=2, marker='o')
                ax1.set_xlabel('Epoch', fontsize=12)
                ax1.set_ylabel('Loss', fontsize=12, color='black')
                ax1.tick_params(axis='y', labelcolor='black')
                ax1.grid(True, alpha=0.3)

                # 创建第二个Y轴（右侧）用于准确率
                ax2 = ax1.twinx()
                # 绘制训练准确率和验证准确率
                ax2.plot(range(1, len(train_accs) + 1), train_accs, 'red', label='train token acc', linewidth=2,
                         linestyle='--')
                ax2.plot(val_epochs, val_accs, 'orange', label='val token acc', linewidth=2, marker='s')
                ax2.set_ylabel('Token Accuracy (%)', fontsize=12, color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                # 设置准确率范围在0-100%
                ax2.set_ylim(0, 100)

                # 合并图例
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')

                plt.title(f'PCB Transformer Training Metrics (Epoch {epoch + 1})', fontsize=14, fontweight='bold')
                plt.tight_layout()
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

    # -------------------------- 初始化Tokenizer --------------------------
    try:
        tokenizer = PCBTokenizer(encode_table_path=config["encode_table_path"])
        print(f"\n成功加载编码表，词汇表大小: {len(tokenizer.encode_table)}")
    except Exception as e:
        print(f"\n初始化Tokenizer失败: {str(e)}")
        return

    # -------------------------- 加载测试数据 --------------------------
    try:
        # 加载测试集
        test_dataset = PCBDataset(data_dir=config["test_dir"], tokenizer=tokenizer, max_length=config["max_length"])

        print(f"\n成功加载测试数据集:")
        print(f"  测试集样本数: {len(test_dataset)}")
    except Exception as e:
        print(f"\n加载测试数据集失败: {str(e)}")
        return

    # -------------------------- 创建测试DataLoader --------------------------
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # -------------------------- 加载模型 --------------------------
    try:
        vocab_size = len(tokenizer.encode_table)
        model = GPTModel(vocab_size, config["d_model"], config["num_layers"], config["nhead"], config["dim_feedforward"], config["dropout"]).to(config["device"])
        # 加载最佳模型权重
        checkpoint = torch.load(config["save_path"], map_location=config["device"])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n成功加载最佳模型（训练轮数: {checkpoint['epoch']}）")
    except Exception as e:
        print(f"\n加载模型失败: {str(e)}")
        return

    # -------------------------- 进行测试 --------------------------
    total_token_acc = 0.0
    test_count = 0

    try:
        for batch in tqdm(test_loader, desc="Testing"):
            test_count += 1
            sample_tokens = batch["tokens"].to(config["device"])
            sample_filename = batch["filename"][0]  # 从列表中获取文件名
            print(f"\n预测示例（测试集样本: {sample_filename}）")

            # 提取提示序列：从<SOS>到<SODPS>
            cond_len = find_cond_length(sample_tokens[0, :])
            if cond_len == -1:
                print(f"样本 {sample_filename} 中未找到<SODPS>标记，无法生成提示")
                continue

            prompt_tokens = sample_tokens[:, :cond_len].to(config["device"])

            # 计算最大生成长度
            max_gen_len = config["max_length"] - prompt_tokens.size(1)
            if max_gen_len <= 0:
                print(f"提示序列长度已达到max_length，无法生成更多token")
                continue

            # 生成布线信息
            generated_tokens = test_model(model, prompt_tokens, sample_tokens, config["device"], max_gen_len, mask=False)

            # 找到EOS位置
            eos_pos_sample = find_token_position(sample_tokens[0, :], tokenizer.special_tokens['<EOS>'])
            eos_pos_gen = find_token_position(generated_tokens[0, :], tokenizer.special_tokens['<EOS>'])

            # 确保索引有效
            end_pos_sample = eos_pos_sample if eos_pos_sample != -1 else config["max_length"]
            end_pos_gen = eos_pos_gen if eos_pos_gen != -1 else config["max_length"]

            # 打印预测结果
            print(f"\n预测结果:")
            print(f"  标签token：{sample_tokens[0, cond_len:end_pos_sample]}")
            print(f"  生成token: {generated_tokens[0, cond_len:end_pos_gen]}")

            # 计算Token正确率
            valid_length = min(end_pos_sample - cond_len, end_pos_gen - cond_len)
            if valid_length <= 0:
                print(f"  无有效预测内容，跳过准确率计算")
                continue

            valid_sample_tokens = sample_tokens[0, cond_len:cond_len + valid_length].cpu()
            valid_gen_tokens = generated_tokens[0, cond_len:cond_len + valid_length].cpu()

            # 计算非PAD token的准确率
            non_pad_mask = (valid_sample_tokens != tokenizer.special_tokens['<PAD>'])
            if non_pad_mask.sum() == 0:
                token_acc = 0.0
            else:
                token_acc = (valid_sample_tokens[non_pad_mask] == valid_gen_tokens[
                    non_pad_mask]).float().mean().item() * 100

            total_token_acc += token_acc
            print(f"  token准确率：{token_acc:.2f}%")

        if test_count > 0:
            avg_token_acc = total_token_acc / test_count
            print(f"\n测试集平均token正确率：{avg_token_acc:.2f}%")
        else:
            print("\n没有有效的测试样本")
    except Exception as e:
        print(f"\n预测示例失败: {str(e)}")
        return

    # -------------------------- 测试完成 --------------------------
    print(f"\n测试流程全部完成！")


if __name__ == "__main__":
    # 可以通过修改这两个标志来控制执行训练还是测试
    train_flag = True
    test_flag = False

    if train_flag:
        train()
    if test_flag:
        test()
