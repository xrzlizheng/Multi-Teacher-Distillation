import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

def load_and_preprocess_data():
    """加载并预处理数据"""
    df = pd.read_csv("data.csv")

    # 将数据分为控制组（训练）和实验组（测试）
    train_df = df[df['PromotionFlag'] == 0]
    test_df = df[df['PromotionFlag'] == 1]

    # 对分类特征进行one-hot编码
    ohe = OneHotEncoder()
    channel_train = ohe.fit_transform(train_df[['PreferredChannel']]).toarray()
    channel_test = ohe.transform(test_df[['PreferredChannel']]).toarray()

    # 创建特征矩阵
    X_train = pd.concat([
        train_df[['Age','Income','DaysSinceLastPurchase','IsHolidaySeason','LoyaltyScore']].reset_index(drop=True),
        pd.DataFrame(channel_train).reset_index(drop=True)
    ], axis=1)
    y_train = train_df['Purchase'].values

    X_test = pd.concat([
        test_df[['Age','Income','DaysSinceLastPurchase','IsHolidaySeason','LoyaltyScore']].reset_index(drop=True),
        pd.DataFrame(channel_test).reset_index(drop=True)
    ], axis=1)
    y_test = test_df['Purchase'].values

    # 确保列名为字符串（用于缩放）
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # 特征缩放
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练-验证集划分
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_tr, X_val, X_test, y_tr, y_val, y_test

# 加载并预处理数据
X_tr, X_val, X_test, y_tr, y_val, y_test = load_and_preprocess_data()

def train_teacher_models(X_train, y_train):
    """训练教师模型"""
    xgb = XGBClassifier().fit(X_train, y_train)
    lgb = LGBMClassifier().fit(X_train, y_train)
    rf = RandomForestClassifier().fit(X_train, y_train)
    return {'XGBoost': xgb, 'LightGBM': lgb, 'RandomForest': rf}

def evaluate_teacher_models(models, X_test, y_test):
    """评估教师模型性能"""
    results = []
    models = teacher_models
    for name, model in models.items():
        pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pred_proba)
        acc = accuracy_score(y_test, pred_proba > 0.5)
        ks_stat, _ = ks_2samp(pred_proba[y_test == 1], pred_proba[y_test == 0])
        
        print(f"{name} Performance:")
        print(f"  - AUC: {auc:.4f}")
        print(f"  - Accuracy: {acc:.4f}")
        print(f"  - KS Statistic: {ks_stat:.4f}\n")
        results.append({'Model': name, 'AUC': auc, 'Accuracy': acc, 'KS Statistic': ks_stat})
    return results

def get_teacher_predictions(models, X):
    """获取教师模型预测结果"""
    return np.vstack([
        models['XGBoost'].predict_proba(X)[:,1],
        models['LightGBM'].predict_proba(X)[:,1],
        models['RandomForest'].predict_proba(X)[:,1]
    ]).T


class MultiTeacherDistillationModel(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.student_head = nn.Sequential(nn.Linear(16,1),nn.Sigmoid())
        self.attention_head = nn.Sequential(nn.Linear(16,3),nn.Softmax(dim=1))
    def forward(self,x):
        f=self.shared(x)
        return self.student_head(f),self.attention_head(f)


def train_distillation_model(X_train, y_train, X_val, y_val, teacher_preds_train, teacher_preds_val, epochs=200, patience=5, lr=0.0005):
    """训练蒸馏模型"""
    model = MultiTeacherDistillationModel(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce_loss = nn.BCELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    teacher_preds_train_t = torch.tensor(teacher_preds_train, dtype=torch.float32)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    teacher_preds_val_t = torch.tensor(teacher_preds_val, dtype=torch.float32)

    best_auc = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        student_pred, att_w = model(X_train_t)
        teacher_loss = torch.mean(torch.sum(att_w*(teacher_preds_train_t-student_pred)**2, dim=1))
        loss = bce_loss(student_pred, y_train_t) + teacher_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(X_val_t)
            val_auc = roc_auc_score(y_val, val_pred.numpy())
            if val_auc > best_auc:
                best_auc = val_auc
                current_patience = patience
                torch.save(model.state_dict(), 'best_model.pt')
            else:
                current_patience -= 1
                if current_patience == 0: break
        print(f"Epoch {epoch+1}: Loss {loss.item():.4f}, Val AUC {val_auc:.4f}")

    model.load_state_dict(torch.load('best_model.pt'))
    return model

def evaluate_distillation_model(model, X_test, y_test):
    """评估蒸馏模型性能"""
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        test_pred, att_w = model(X_test_t)
        test_pred_np = test_pred.numpy()
        if test_pred_np.ndim > 1:
            test_pred_np = test_pred_np.flatten()
        auc = roc_auc_score(y_test, test_pred_np)
        acc = accuracy_score(y_test, test_pred_np > 0.5)
        ks_stat, _ = ks_2samp(test_pred_np[y_test == 1], test_pred_np[y_test == 0])
        att_mean = att_w.numpy().mean(axis=0)

    print(f"\nTest AUC: {auc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test KS: {ks_stat:.4f}")
    print(f"Mean Attention Weights (XGB,LGB,RF): {att_mean}")

    return {'AUC': auc, 'Accuracy': acc, 'KS Statistic': ks_stat, 'Attention Weights': att_mean}



def _plot_bar_chart(ax, labels, values, ylabel, title, color):
    """绘制柱状图并添加数值标签的辅助函数"""
    bars = ax.bar(labels, values, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}',
               ha='center', va='bottom')

def plot_teacher_vs_student(teacher_results, student_results):
    """展示教师模型和学生模型的性能"""
    # 提取教师模型和学生模型的性能指标
    metrics = {
        'AUC': [result['AUC'] for result in teacher_results] + [student_results['AUC']],
        'Accuracy': [result['Accuracy'] for result in teacher_results] + [student_results['Accuracy']],
        'KS Statistic': [result['KS Statistic'] for result in teacher_results] + [student_results['KS Statistic']]
    }

    # 设置图表基本参数
    labels = [result['Model'] for result in teacher_results] + ['Distilled']

    # 创建包含三个独立图表的布局
    fig = plt.figure(figsize=(18, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 定义颜色配置
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # 绘制三个子图
    _plot_bar_chart(fig.add_subplot(131), labels, metrics['AUC'], 'AUC', '模型AUC表现', colors)
    _plot_bar_chart(fig.add_subplot(132), labels, metrics['Accuracy'], '准确率', '模型准确率表现', colors)
    _plot_bar_chart(fig.add_subplot(133), labels, metrics['KS Statistic'], 'KS统计量', '模型KS统计量表现', colors)

    # 调整布局并保存
    fig.tight_layout()
    plt.savefig('teacher_vs_student_bar_chart.png')
    plt.show()

# 训练和评估教师模型
teacher_models = train_teacher_models(X_tr, y_tr)
teacher_results = evaluate_teacher_models(teacher_models, X_test, y_test)
tp_train, tp_val, tp_test = get_teacher_predictions(teacher_models, X_tr), get_teacher_predictions(teacher_models, X_val), get_teacher_predictions(teacher_models, X_test)

# PyTorch Data
X_tr_t, y_tr_t = torch.tensor(X_tr,dtype=torch.float32), torch.tensor(y_tr,dtype=torch.float32).unsqueeze(1)
tp_train_t = torch.tensor(tp_train,dtype=torch.float32)

X_val_t, y_val_t = torch.tensor(X_val,dtype=torch.float32), torch.tensor(y_val,dtype=torch.float32).unsqueeze(1)
tp_val_t = torch.tensor(tp_val,dtype=torch.float32)

X_test_t = torch.tensor(X_test,dtype=torch.float32)

# 训练和评估蒸馏模型
model = train_distillation_model(X_tr, y_tr, X_val, y_val, tp_train, tp_val)
distillation_results = evaluate_distillation_model(model, X_test, y_test)
plot_teacher_vs_student(teacher_results, distillation_results)