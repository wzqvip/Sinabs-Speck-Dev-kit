# %%
#%pip install -r ./requirements.txt

# %%
from tonic.datasets.nmnist import NMNIST
from tonic.transforms import ToFrame

root_dir = "./NMNIST"
NMNIST(save_to=root_dir, train=True)
NMNIST(save_to=root_dir, train=False)

to_raster = ToFrame(sensor_size=(34, 34), n_time_bins=100)
dataset = NMNIST(save_to=root_dir, train=False, transform=to_raster)


# %%
from torch import nn

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


# %%
from sinabs.from_torch import from_model

snn_convert = from_model(model=cnn, input_shape=(2, 34, 34), batch_size=4).spiking_model


# %%
from sinabs.backend.dynapcnn import DynapcnnNetwork

devkit_name = "speck2fdevkit"

cpu_snn = snn_convert.to("cpu")
dynapcnn = DynapcnnNetwork(snn=cpu_snn, input_shape=(2, 34, 34), discretize=True, dvs_input=False)
# dynapcnn.to(device="speck2fdevkit", chip_layers_ordering="auto")

dynapcnn.to(device=devkit_name, chip_layers_ordering="auto", monitor_layers=["dvs", 3])



# %%
from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer

visualizer = DynapcnnVisualizer(window_scale=(4, 8), dvs_shape=(34, 34), spike_collection_interval=50)
visualizer.connect(dynapcnn)


# %%
dvs_event = samna.speck2f.event.DvsEvent()
# 设置 dvs_event 的属性：x, y, t, p
# 然后发送到 devkit 推理
output = dynapcnn([dvs_event])


# %%
import samna
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork

# 打开设备
devices = samna.device.get_unopened_devices()
if len(devices) == 0:
    raise Exception("No devices found")

# 打开第一个设备
my_board = samna.device.open_device(devices[0])

# 获取设备的 serial_number
device_id = my_board.get_serial_number()  # 获取设备的 serial_number 作为设备 ID

# 你的 SNN 模型，例如通过 CNN -> SNN 转换得到
snn_model = from_model(
    model=cnn,  # 你的 CNN 模型
    input_shape=(2, 34, 34),
    batch_size=4
).spiking_model

# 包装为 DynapcnnNetwork
dynapcnn = DynapcnnNetwork(
    snn=snn_model.to("cpu"),
    input_shape=(2, 34, 34),
    discretize=True,
    dvs_input=False
)

# ✅ 使用 serial_number 作为设备标识符
dynapcnn.to(device=device_id, chip_layers_ordering="auto")

print(f"✅ 模型成功部署到 SPECK！部署到核心顺序为: {dynapcnn.chip_layers_ordering}")


# %%
from tonic.datasets.nmnist import NMNIST
from tonic.transforms import ToFrame
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm.notebook import tqdm
from torch.nn import CrossEntropyLoss
import samna
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer
from collections import Counter
import torch

# 数据下载并准备
root_dir = "./NMNIST"
_ = NMNIST(save_to=root_dir, train=True)
_ = NMNIST(save_to=root_dir, train=False)

# 数据预处理
to_frame = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=1)
cnn_train_dataset = NMNIST(save_to=root_dir, train=True, transform=to_frame)
cnn_test_dataset = NMNIST(save_to=root_dir, train=False, transform=to_frame)

# 定义CNN模型
cnn = nn.Sequential(
    nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 3), padding=(1, 1), bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=(1, 1), bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2),  bias=False),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 10, bias=False),
    nn.ReLU(),
)

# 初始化CNN权重
for layer in cnn.modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(layer.weight.data)

# 训练CNN
epochs = 1
lr = 1e-3
batch_size = 4
device = "cpu"
cnn = cnn.to(device=device)
cnn_train_dataloader = DataLoader(cnn_train_dataset, batch_size=batch_size, shuffle=True)
cnn_test_dataloader = DataLoader(cnn_test_dataset, batch_size=batch_size, shuffle=False)

optimizer = SGD(params=cnn.parameters(), lr=lr)
criterion = CrossEntropyLoss()

for e in range(epochs):
    train_p_bar = tqdm(cnn_train_dataloader)
    for data, label in train_p_bar:
        data = data.squeeze(dim=1).to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.long)
        optimizer.zero_grad()
        output = cnn(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_p_bar.set_description(f"Epoch {e} - Training Loss: {round(loss.item(), 4)}")

    # 验证CNN
    correct_predictions = []
    with torch.no_grad():
        test_p_bar = tqdm(cnn_test_dataloader)
        for data, label in test_p_bar:
            data = data.squeeze(dim=1).to(device=device, dtype=torch.float)
            label = label.to(device=device, dtype=torch.long)
            output = cnn(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct_predictions.append(pred.eq(label.view_as(pred)))
        correct_predictions = torch.cat(correct_predictions)
        print(f"Epoch {e} - accuracy: {correct_predictions.sum().item() / len(correct_predictions) * 100}%")

# CNN转SNN
snn_convert = from_model(model=cnn, input_shape=(2, 34, 34), batch_size=batch_size).spiking_model

# 定义SNN测试数据集
n_time_steps = 100
to_raster = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=n_time_steps)
snn_test_dataset = NMNIST(save_to=root_dir, train=False, transform=to_raster)
snn_test_dataloader = DataLoader(snn_test_dataset, batch_size=batch_size, shuffle=False)

# 测试SNN
snn_convert = snn_convert.to(device)
correct_predictions = []
with torch.no_grad():
    test_p_bar = tqdm(snn_test_dataloader)
    for data, label in test_p_bar:
        data = data.reshape(-1, 2, 34, 34).to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.long)
        output = snn_convert(data)
        output = output.reshape(batch_size, n_time_steps, -1)
        output = output.sum(dim=1)
        pred = output.argmax(dim=1, keepdim=True)
        correct_predictions.append(pred.eq(label.view_as(pred)))
    correct_predictions = torch.cat(correct_predictions)
    print(f"accuracy of converted SNN: {correct_predictions.sum().item() / len(correct_predictions) * 100}%")

# 将SNN部署到Devkit
cpu_snn = snn_convert.to(device="cpu")
dynapcnn = DynapcnnNetwork(snn=cpu_snn, input_shape=(2, 34, 34), discretize=True, dvs_input=False)
devkit_name = "speck2fmodule"
dynapcnn.to(device=devkit_name, chip_layers_ordering="auto")
print(f"The SNN is deployed on the core: {dynapcnn.chip_layers_ordering}")

# 生成事件流并进行推理
snn_test_dataset = NMNIST(save_to=root_dir, train=False)
subset_indices = list(range(0, len(snn_test_dataset), 100))
snn_test_dataset = Subset(snn_test_dataset, subset_indices)

inferece_p_bar = tqdm(snn_test_dataset)
test_samples = 0
correct_samples = 0
for events, label in inferece_p_bar:
    samna_event_stream = []
    for ev in events:
        spk = samna.speck2f.event.Spike()
        spk.x = ev['x']
        spk.y = ev['y']
        spk.timestamp = ev['t'] - events['t'][0]
        spk.feature = ev['p']
        spk.layer = 0
        samna_event_stream.append(spk)

    output_events = dynapcnn(samna_event_stream)
    neuron_index = [each.feature for each in output_events]
    if len(neuron_index) != 0:
        frequent_counter = Counter(neuron_index)
        prediction = frequent_counter.most_common(1)[0][0]
    else:
        prediction = -1
    inferece_p_bar.set_description(f"label: {label}, prediction: {prediction}, output spikes num: {len(output_events)}")

    if prediction == label:
        correct_samples += 1
    test_samples += 1
print(f"On chip inference accuracy: {correct_samples / test_samples}")

# 可视化SNN部署
visualizer = DynapcnnVisualizer(
    window_scale=(4, 8),
    dvs_shape=(34, 34),
    spike_collection_interval=50,
)

visualizer.connect(dynapcnn)



