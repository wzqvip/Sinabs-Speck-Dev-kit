# Step 1: 安装依赖库（请确保这些已经在环境中安装好）
# %pip install -r ./requirements.txt

# Step 2: 导入必要的库和模块
from tonic.datasets.nmnist import NMNIST
from tonic.transforms import ToFrame
import torch
from torch import nn
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork
import samna
from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer

# Step 3: 加载并转换 N-MNIST 数据集
root_dir = "./NMNIST"

# 下载训练和测试数据集
NMNIST(save_to=root_dir, train=True)
NMNIST(save_to=root_dir, train=False)

# 选择一部分数据集进行训练（这里只取前100个样本）
to_raster = ToFrame(sensor_size=(34, 34,1), n_time_bins=100)
dataset = NMNIST(save_to=root_dir, train=True, transform=to_raster)
train_subset = torch.utils.data.Subset(dataset, range(100))  # 只取前100个样本

# 数据加载器
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=4, shuffle=True)

# Step 4: 定义 CNN 模型
cnn = nn.Sequential(
    nn.Conv2d(2, 8, 3, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2),
    nn.Conv2d(16, 16, 3, padding=1, stride=2, bias=False),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(256, 10, bias=False),
    nn.ReLU()
)

# Step 5: 训练 CNN 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

# 训练简单的 CNN
num_epochs = 5
for epoch in range(num_epochs):
    cnn.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = cnn(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Step 6: 将训练好的 CNN 模型转换为 SNN
snn_convert = from_model(model=cnn, input_shape=(2, 34, 34), batch_size=4).spiking_model

# Step 7: 部署到 Devkit
devices = samna.device.get_unopened_devices()
if len(devices) == 0:
    raise Exception("No devices found")

my_board = samna.device.open_device(devices[0])
device_id = my_board.get_serial_number()

# 使用 DynapcnnNetwork 将模型部署到设备
cpu_snn = snn_convert.to("cpu")
dynapcnn = DynapcnnNetwork(snn=cpu_snn, input_shape=(2, 34, 34), discretize=True, dvs_input=False)
dynapcnn.to(device=device_id, chip_layers_ordering="auto")

print(f"✅ 模型成功部署到 SPECK！部署到核心顺序为: {dynapcnn.chip_layers_ordering}")

# Step 8: 使用 DynapcnnVisualizer 监控推理过程
visualizer = DynapcnnVisualizer(window_scale=(4, 8), dvs_shape=(34, 34), spike_collection_interval=50)
visualizer.connect(dynapcnn)

# Step 9: 进行推理（此时需要将事件传递给 Devkit）
dvs_event = samna.speck2f.event.DvsEvent()
dvs_event.x = 10  # x 坐标
dvs_event.y = 20  # y 坐标
dvs_event.t = 50  # 时间戳（可以是一个步长）
dvs_event.p = 1   # 极性（1 或 0）

output = dynapcnn([dvs_event])  # 推理结果
print("推理输出：", output)
