import torch
import torch.nn as nn
import time


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # 残差连接
        return self.relu(out)


class DeepResNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, num_blocks=8):
        super().__init__()
        
        # 输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 残差块
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(ResidualBlock(hidden_dim))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.output_proj(x)
        return x


class DeepMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 1024, 2048, 1024, 512]):
        super().__init__()
        
        layers = []
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # 构建隐藏层
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)


class ModelComparison:
    def __init__(self, input_dim, output_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 初始化两个模型
        self.mlp = DeepMLP(input_dim, output_dim).to(device)
        self.resnet = DeepResNet(input_dim, output_dim).to(device)
        
        # 设置相同的优化器和损失函数
        self.criterion = nn.MSELoss()
        self.mlp_optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=0.001, weight_decay=0.01)
        self.resnet_optimizer = torch.optim.AdamW(self.resnet.parameters(), lr=0.001, weight_decay=0.01)
        
    def train_epoch(self, model, optimizer, train_loader):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = self.criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def evaluate(self, model, val_loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                loss = self.criterion(output, y)
                total_loss += loss.item()
        return total_loss / len(val_loader)
    
    def compare_models(self, train_loader, val_loader, epochs=100):
        results = {
            'mlp': {'train_loss': [], 'val_loss': [], 'time': 0},
            'resnet': {'train_loss': [], 'val_loss': [], 'time': 0}
        }
        
        # 训练MLP
        print("Training MLP...")
        start_time = time.time()
        for epoch in range(epochs):
            train_loss = self.train_epoch(self.mlp, self.mlp_optimizer, train_loader)
            val_loss = self.evaluate(self.mlp, val_loader)
            results['mlp']['train_loss'].append(train_loss)
            results['mlp']['val_loss'].append(val_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        results['mlp']['time'] = time.time() - start_time
        
        # 训练ResNet
        print("\nTraining ResNet...")
        start_time = time.time()
        for epoch in range(epochs):
            train_loss = self.train_epoch(self.resnet, self.resnet_optimizer, train_loader)
            val_loss = self.evaluate(self.resnet, val_loader)
            results['resnet']['train_loss'].append(train_loss)
            results['resnet']['val_loss'].append(val_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        results['resnet']['time'] = time.time() - start_time
        
        return results


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    input_dim = 10
    output_dim = 1
    batch_size = 32
    
    # 生成一些随机数据用于测试
    X = torch.randn(1000, input_dim)
    y = torch.sum(X * torch.randn(input_dim), dim=1, keepdim=True)
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # 进行模型比较
    comparison = ModelComparison(input_dim, output_dim)
    results = comparison.compare_models(train_loader, val_loader, epochs=50)
    
    # 打印结果
    print("\nResults Summary:")
    print(f"MLP Training Time: {results['mlp']['time']:.2f} seconds")
    print(f"MLP Final Val Loss: {results['mlp']['val_loss'][-1]:.6f}")
    print(f"ResNet Training Time: {results['resnet']['time']:.2f} seconds")
    print(f"ResNet Final Val Loss: {results['resnet']['val_loss'][-1]:.6f}")