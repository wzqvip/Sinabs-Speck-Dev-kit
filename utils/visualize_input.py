import samna
import sinabs.backend.dynapcnn.io as sio

# ------------------------
# 1. 打开 Speck2e 开发板
# ------------------------

device_map = sio.get_device_map()
print("Detected Devices:", device_map)



# 这里我们默认使用第一个 speck2e 开发板
device_id = list(device_map.keys())[0]  # e.g. 'speck2edevkit:0'
devkit = sio.open_device(device_id)

# ------------------------
# 2. 构建 Samna Graph
# ------------------------

# 创建事件处理图
samna_graph = samna.graph.EventFilterGraph()

# 连接事件流：设备 -> 可视化转换器 -> 可视化输出流
# 替换为 Speck2f 的事件转换器
_, _, streamer = samna_graph.sequential([
    devkit.get_model_source_node(),       # 从 devkit 读取事件
    "Speck2fDvsToVizConverter",           # ⚠️ 这里改了！
    "VizEventStreamer"                    # 发送到 GUI 窗口
])


# ------------------------
# 3. 启动可视化窗口
# ------------------------

visualizer_port = "tcp://0.0.0.0:40000"
gui_process = sio.launch_visualizer(receiver_endpoint=visualizer_port, disjoint_process=True)

# ------------------------
# 3.1 配置可视化视图
# ------------------------

visualizer_config, _ = samna_graph.sequential([
    samna.BasicSourceNode_ui_event(),  # 生成 UI 控制指令
    streamer                            # 将配置流入 GUI
])

# 设置连接目标端口
streamer.set_streamer_destination(visualizer_port)

# 确保连接成功
if streamer.wait_for_receiver_count() == 0:
    raise Exception(f"连接可视化器失败（端口: {visualizer_port}）")

# 配置视图：一个 128x128 的 DVS 活动图
plot1 = samna.ui.ActivityPlotConfiguration(
    image_width=128,
    image_height=128,
    title="DVS Layer",
    layout=[0, 0, 1, 1]
)

# 写入视图配置
visualizer_config.write([
    samna.ui.VisualizerConfiguration(plots=[plot1])
])

# ------------------------
# 4. 启动图并启用 DVS 可视化
# ------------------------

samna_graph.start()

# Speck2f 的配置对象
from samna.speck2f.configuration import SpeckConfiguration
devkit_config = SpeckConfiguration()

# 启用原始 DVS 数据监控
devkit_config.dvs_layer.raw_monitor_enable = True
devkit.get_model().apply_configuration(devkit_config)


print("✅ DVS 可视化已启动，打开的窗口应显示实时活动。")

# ------------------------
# 可视化运行中，此脚本阻塞运行；可添加 input() 或 sleep 来保持状态
# ------------------------
input("按回车键停止可视化...")

# ------------------------
# 5. 停止图与可视化窗口
# ------------------------

samna_graph.stop()

# 关闭 GUI 可视化窗口（如是子进程启动）
if gui_process:
    gui_process.terminate()
    gui_process.join()

print("✅ 可视化已停止，资源已释放。")
