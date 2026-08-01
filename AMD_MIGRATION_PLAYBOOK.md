# AMD Migration Playbook

这份文档记录 Datawhale-EAI 在 AMD Radeon + ROCm 上迁移具身智能项目时的
可复现边界。它把“能导入”“能渲染”“能训练”“能闭环评估”分成不同门禁，
避免把单个 smoke demo 当成整仓可用。

配套长文：[AMD 具身智能迁移实战博客](migration-blog.html)。博客按
DISCOVERSE、RoboCasa365、DexJoCo/JAX 和策略模型逐项解释问题、修复顺序与
结果边界；本手册保留可执行的短版门禁。

## 统一验收门禁

1. **环境**：ROCm、PyTorch/JAX、MuJoCo/Genesis/DISCOVERSE 依赖可导入。
2. **仿真**：官方场景能 reset、step、render，并能保存 MP4。
3. **数据**：专家轨迹或官方数据能生成/读取，维度、时间顺序和元数据通过审计。
4. **训练**：官方入口能启动，日志、checkpoint、断点续训和磁盘路径可复现。
5. **推理**：使用同一 checkpoint、归一化统计和官方 observation/action bridge。
6. **评估**：固定任务与 seeds，保留成功/失败视频和 JSON；成功率只来自完整分母。

## 已验证迁移

### ROCm JAX / OpenPI

- 官方安装入口：[AMD ROCm JAX 0.10.0 安装指南](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html?fam=all&os=linux&jax-ver=0.10.0&i=docker&w=compute)；推荐镜像为 `rocm/jax:rocm7.14-jax0.10.0-py3.12`。
- RoboCasa365 的官方 Pi0.5 路线已用原生 JAX 验证：GPU backend、75k checkpoint、tokenizer、归一化统计、Gemma softmax 隔离路径和 OSMesa 视频输出均已通过；一个真实 seed 的诊断 rollout 为 `1/1`，不作为正式成绩。
- 同一 `16 tasks x 50 episodes` 正式协议的 Pi0.5 结果为 `142/800 = 17.75%`，对应 JSON、每任务统计、视频和 SHA 已归档。
- DexJoCo 必须单独看：当前旧主机环境是 ROCm 7.2.1/JAX 0.8.2，GPU 可见但 GEMM 报 `hipGetFuncBySymbol`。已新增官方容器 preflight，先验证 JAX 矩阵计算，再做固定任务/seed 对照；在此之前不把 JAX 转 PyTorch 的桥接结果写成 JAX 成绩。
- 结论：JAX 0.10.0 可以作为新的直接迁移候选，但不能只改 Python 包版本；必须保持官方容器内 ROCm、JAX plugin、PJRT 和 Python 版本一致。

### DISCOVERSE

- 上游：[DISCOVERSE](https://github.com/discoverse-dev/DISCOVERSE)。
- 本地迁移工作区：私有 AMD migration workspace（路径不公开）。
- 已验证：核心运行 smoke `18/18`、AIRBOT `12/12`、MMK2 `8/8`、
  `block_bridge_place` 严格专家回放 `31/31`、专家数据 `500 episodes`。
- 覆盖：MuJoCo 任务、专家轨迹、训练/推理入口、MP4 输出、ROS2、LiDAR、
  3DGS 以及 ACT/DP/PPO/RDT 运行链路。
- 边界：这些数字证明迁移和专家/运行链路，不等于学习策略已经成功；严格闭环
  负结果继续保留。真实设备、RealSense、手柄和 ROS2 外设仍需对应硬件。

### RoboCasa365

- AMD 395 已完成官方资产、场景 reset/render 和视频输出。
- Pi0.5 与 GR00T 已按同一 `16 tasks x 50 episodes` 视频协议完成匹配评估：
  Pi0.5 `142/800`，GR00T `230/800`。
- 正式结果 JSON 与 SHA 见成果站和 `WORKSPACE_MEMORY.md`；随机策略 smoke 不计入模型成绩。
- 独立 showcase 录像：AMD 395 上真实 `CloseFridge` 两回合为 `1/2`，center/left/right/eye_in_hand 四视角合成 `1920x1080@20fps`；只提升可视化，不改变策略输入或正式分母。
- 边界：Pi0.5 ROCm/JAX 的官方兼容路径已单独隔离，不能用短 smoke 替代正式结果。

### Every Embodied VLA

- SmolVLA、Pi0、ACT 的普通训练、保护训练、Notebook、推理和严格评估入口已整理。
- 当前可复核结果：SmolVLA `57/60`、Pi0 `12/14`、ACT 诊断线 `7/30`。
- 保护权重、评估 JSON、视频和 SHA 必须绑定到同一 Notebook 训练产物，不能把历史
  权重冒充当前 Notebook 的结果。

### RoboWits

- W7900 上使用官方 16-D ACT 配置进行 `100k` 训练，checkpoint 每 `5k` 备份到
  Hugging Face；官方闭环成绩在完整评估前不提前声明。
- 云端 `/workspace` 是 PVC 持久目录；系统包、SSH 服务和动态端口可能随实例重建，
  所以训练/同步脚本必须放在 PVC 并记录当前端点。

## AMD 适配中最容易出错的地方

| 层级 | 典型问题 | 处理原则 |
|---|---|---|
| 设备 | `torch.cuda` API 名称仍存在，但实际后端是 HIP | 同时记录 `torch.version.hip`、`rocminfo`、GPU 型号 |
| 渲染 | Vulkan/CUDA denoiser 告警、无窗口环境 | 将 offscreen/MP4 作为一等输出，告警与任务失败分开 |
| 数据 | action chunk、episode 边界、归一化统计错位 | 先做时间顺序、维度、终端 padding 和 stats 审计 |
| JAX | ROCm wheel、MIOpen、attention/shared-memory 限制 | 记录版本与算子失败；不把 PyTorch 结果移作 JAX 结果 |
| 资产 | BlenderKit/专属资产不可公开或缺失 | 保持官方分母，单独标记 blocked，不替换场景 |
| 评估 | 冷启动/热复用、视频路径、并行拓扑改变 RNG | 固定官方任务顺序、进程拓扑、seeds 和输出协议 |

## 发布边界

- 公开站只放脱敏后的结果、视频、配置摘要和上游链接。
- 私有比赛仓库保存 AMD bridge、补丁、资产清单和详细日志。
- 不上传 token、机器地址、个人路径、原始私有资产或未经授权的代理网格。
- 每个 checkpoint、JSON 和视频都要有 SHA256；训练进度和失败原因写入工作区记忆。

## DexJoCo / Pi0.5

公开检索到的项目是 [DexJoCo](https://dexjoco.github.io/)，代码入口为
[brave-eai/dexjoco](https://github.com/brave-eai/dexjoco)，策略权重入口为
[DexJoCo-Pi05](https://huggingface.co/DexJoCo/DexJoCo-Pi05)。AMD ROCm JAX
0.10.0 环境已完成 GPU 预检和原生 Orbax restore；`water_plant` 诊断为
`4/4`，官方固定 seed 的 11 任务评估为 `5/11`。这两个数字回答不同问题，不能
互相替代；多任务恢复搜索中有 10/11 任务找到至少一个成功 seed，仍保留原始
官方分母和失败任务。
