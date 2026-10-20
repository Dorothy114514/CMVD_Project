# CMVD PET/MR 研究项目

已实现D1只读影像读取、D2桌面浏览、D3人工ROI/原始曲线、D4探索性灌注特征/PET工作站结果适配和D5队列校验/患者划分工程入口，以及独立的合成推理框架；尚未训练真实诊断模型。真实清单、临床/诊断表和特征审核记录尚缺，当前验证范围见交接。

先阅读 [项目状态与交接](docs/HANDOFF.md)；AI协作规则见 [AGENTS.md](AGENTS.md)。

| 位置 | 用途 |
|---|---|
| data/ | 已放入原始资料；31/48个7z及阳性/阴性各1个解压样例，只读 |
| deidentified/ | 已处理影像约定位置；历史79例，当前未迁入，仍需像素及完整隐私审查 |
| metadata/full_case_manifest.csv | 病例索引与处理计数 |
| audit/ | 原有规则复核和排除记录 |
| audit/restricted/ | 含真实患者文件名的完整性审计 |
| scripts/data_preparation/ | 数据处理与复核脚本 |
| scripts/data_inspection/preview_samples.py | 仅检查两例已解压样例并本地渲染少量帧，不打开压缩包 |
| tools/7zr.exe | 本地解压工具 |
| docs/HANDOFF.md | 当前进展、局限、后续工作及精简日志 |
| docs/DEMO_DESIGN.md | 任务书对照、初步调研、框架方案与年底里程碑 |
| configs/demo_contract.json | 三类输入及PET/MR双灌注的候选字段合同 |
| src/cmvd_demo/ | 只读影像、桌面、ROI/原始曲线与独立合成推理模块 |
| scripts/data_inspection/validate_reader.py | D1两例开发样本读取回归，独立输出报告 |
| scripts/demo/run_demo.py | 仅支持合成病例的命令行骨架 |
| scripts/desktop/start.ps1 | 从任意工作目录启动本机影像浏览器 |
| scripts/desktop/check_desktop.py | 原生Qt自动交互回归；可选两例开发样本 |
| src/cmvd_demo/desktop/ | 桌面窗口、后台任务与有界单帧缓存 |
| src/cmvd_demo/roi/ | 无Qt依赖的ROI几何、原始曲线、来源绑定与快照存储 |
| src/cmvd_demo/perfusion/ | 无Qt依赖的MR逐平面特征、PET工作站JSON适配、参考比较与特征快照 |
| scripts/desktop/check_perfusion.py | D4原生合成流程及复用D3受控曲线的限定回归，不扫描影像 |
| src/cmvd_demo/cohort/ | 无Qt的D5关联校验、D4快照复算、开发矩阵、患者划分与冻结 |
| configs/cohort_contract.json | D5六项候选临床字段及两项灌注比值的工程合同 |
| configs/cohort_project.template.json | D5输入位置及待确认策略；不是已审核项目配置 |
| scripts/demo/prepare_cohort.py | D5队列校验/空白表/完全合成队列CLI，不扫描影像 |
| scripts/desktop/check_cohort.py | D5原生Qt合成检查与真实资料可用性审计 |
| examples/synthetic/ | 完全虚构的病例与未训练演示系数 |
| tests/ | 合同、DICOM读取、ROI数值与Qt桌面交互测试 |
| configs/clinical_candidate_fields.json | 候选临床字段、单位及缺失值说明 |
| examples/templates/ | 仅表头的临床/诊断及D5补录模板，无真实病例 |
| configs/runtime.json | 本地解释器版本、官方安装包与校验值 |
| configs/runtime.local.json | 自动生成的本机环境路径，不进Git |
| requirements/ | base.txt直接依赖及windows-py312.lock完整wheel哈希锁 |
| scripts/setup/ | 环境准备与合成数据环境自检 |
| scripts/python.ps1 | 统一Python启动入口，无需激活或修改PATH |

目标：2026-12-31前完成1个融合临床危险因素、临床特征和PET/MR双灌注图像特征的CMVD阴性/阳性模型，并在≥1例独立测试病例演示完整流程。详见[设计稿](docs/DEMO_DESIGN.md)。

用户已确认标签来源为医院反馈的医生诊断，目前无临床结构表；候选字段以任务书p11为基础，单位、吸烟史编码及采集时点待与医院核对。后续受控采集沿用原研究患者ID，真实数据存入被Git忽略的`metadata/`；不要填写到`examples/`或把未知值当成阴性/零。标签单独存表，`cmvd_label`约定0=医生诊断阴性、1=医生诊断阳性，未确定留空，并保留诊断依据版本及来源引用。模板不会被当前合成框架自动导入。

用户已选择本机桌面应用。已使用PySide6+PyQtGraph实现D2浏览界面，pydicom/NumPy组装空间时间轴；后续SimpleITK/SciPy处理、scikit-learn融合建模，细节及样本约束见[设计稿第9节](docs/DEMO_DESIGN.md#9-根据两例实际样本收敛技术栈)。SimpleITK尚未安装。

启动桌面浏览器（默认空工作区）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\desktop\start.ps1
```

可加`--synthetic`打开完全虚构的方向体模，或`--case-dir '单个已解压病例目录的绝对路径'`；也可在窗口点击“打开病例目录”，选择阳性/阴性目录下面的单个病例文件夹。原件只读，压缩包不会被打开，界面不显示姓名或原始UID。数据用途/静息/负荷均为候选，需人工核对。

左侧选择PET、MR逐平面灌注或cine；下方滑块控制时间/层面，窗宽/窗位只改变显示。浏览帧率可设1–30 fps（默认15），播放等待当前帧完成Qt绘制后推进，慢读取时实际速度会降低；这不是实际采集速度。快拖时保留上一帧及其时间/层面标签，状态栏显示目标，松手立即请求最终位置；拖动会暂停播放。双击序列或按回车可重试，支持后台加载与取消。鼠标滚轮缩放、中键平移；左键行为由图像上方“鼠标工具”选择。方向为原始采集平面，非强制标准轴位。当前明确显示“模型未配置”，没有真实诊断。浏览约定见[设计稿第12节](docs/DEMO_DESIGN.md#12-d2桌面浏览)。

D3使用右侧“ROI / 原始曲线”面板；关闭面板后可从“视图”菜单恢复。已有旧窗口需关闭后重启：

1. 选择序列和层面，在合适的参考帧选择“心肌”并点击“放置参考轮廓”，自动进入“编辑ROI控制点”；左键拖动放大的控制点，点偏到背景不会拖图或拖动整个ROI。再选择“血池”放置并调整。初始轮廓只是几何占位，不是自动分割；内外轮廓可分别调整，点击线段增加顶点，右键控制点可删除顶点（至少保留3点）。
2. 切换时间帧逐帧检查，必要时选择“仅修正当前帧”修改轮廓；可用“取消本帧修正”恢复参考。参考轮廓用于所有未单独修正帧，修正之间不插值；已有独立轮廓的帧应在当前帧模式编辑。
3. 点击“计算原始曲线”，读取当前层的完整时间序列，显示心肌/血池均值和当前帧像素数、面积。PET保留BQML及帧长，MR保留原始信号与非等间隔时间，cine保留当前层实际触发时间。静态图像可以保存ROI，不能凭空生成时间曲线。
4. 点击“保存ROI / 曲线”，在`outputs/roi/<时间戳-随机ID>/`新建快照；“重载ROI”选择其中的`roi.json`。ROI覆盖该序列所有已标注层，`curves.json`/`curves.csv`仅包含当前计算层；来源、几何或时间不匹配会拒绝加载。换病例前请保存，跨病例不保留未保存标注。

需要整体平移某个轮廓时，上方鼠标工具选“移动整个ROI”，左键拖动当前所选心肌/血池的轮廓内部或边线，心肌内外膜一起平移；要左键拖图则选“平移图像”，此时ROI锁定。任何工具下均可中键平移、滚轮缩放，误移视图可点“适应视图”。

图像上方和ROI页均有“撤销/重做”：`Ctrl+Z`撤销，`Ctrl+Y`或`Ctrl+Shift+Z`重做，支持拖点/整体移动、增删顶点、放置/删除ROI、内膜开关、本帧覆盖/恢复及ROI重载。一次拖动只计一步；无效编辑可撤销回上次有效轮廓。历史按序列隔离，当前病例合计最多50步/16 MiB，切回同序列可继续使用，换病例或退出清空；输入框仍使用自身的文本撤销。撤销后只恢复轮廓和编辑帧，不恢复复核勾选、曲线或保留特征，需重新复核/计算；历史不能代替保存快照。

ROI越界、自交、空区域或心肌/血池重叠会阻断相应计算。轮廓修改会使旧曲线与人工复核状态失效。仅在实际核对后勾选方向/覆盖和轮廓/运动复核；它们是本地人工确认，不是临床定量验证。当前运动处理是**逐帧手工修正ROI**，不做自动配准、图像重采样或MBF/MFR计算；真实测量与独立查看器人工核对仍待完成。数值和存储约定见[设计稿第13节](docs/DEMO_DESIGN.md#13-d3人工roi与原始曲线)。

D4在同一右侧面板的“D4 灌注特征”页，已有窗口需重启：

1. MR：先在ROI页计算双ROI曲线；在D4页手动确认静息/负荷，填写无身份信息的协议代码，按曲线选择基线起始帧、基线帧数及首过起止帧（界面均从1计）。基线至少2帧且必须在分析窗口前；可跳过初始信号过渡，但不能自动挑选使结果通过的窗口。计算峰值、增强、窗口起点到达峰时间、相邻采样最大上升斜率及血池归一化斜率。
2. 分别“保留”静息和负荷MR结果，再核对同一解剖平面和剂量/序列/协议可比较性。当前一次仅配对一个平面，MPRI为两状态归一化斜率之比；不自动把长轴/短轴或不同层面平均，也不是绝对MBF。归一化斜率采用心肌/血池斜率比，单位为`1`，与旧合成输入中`s^-1`示例字段不自动互换。
3. PET：在ROI页分别计算两状态动态PET双ROI曲线，D4页手动选择状态并保留。导出PET填写模板后，将已审核工作站的全左室MBF、单位、方法/软件版本、协议/示踪剂、参考和审核者代码填入JSON，再导入。模板MBF默认空，不得为真实病例填演示数字；当前仅支持`mL/min/g`，不是从BQML或TAC自行算MBF。导入核对本地来源、ROI及曲线签名；这不等于独立验证工作站MBF数值。
4. 可导出MR参考模板，填写独立参考值及事先约定的工程容差后核对；模板不预填参考值或容差。保存特征/QC快照至`outputs/features/<新目录>/`，包含JSON/CSV、原始输入/方法/单位/来源和完成标记；无PET或MR参考时可保存已配对MR预览，缺项明确标记。ROI修改会使相关保留结果失效；来源文件变化会拒绝导入/导出。当前不会向合成模型传入这些结果，也不生成诊断。

只演示D4可从顶部“合成演示→D4首过灌注样例（完全合成）”加载两状态，使用基线第1–2帧、首过第3–7帧及同一虚构协议代码走MR流程；不需要勾选未完成的审核。原“合成体模”按钮仍打开方向/播放亮环，不是首过灌注样例。两种合成资料均不可成为已审核真实特征。计算公式与JSON合同见[设计稿第14节](docs/DEMO_DESIGN.md#14-d4探索性双灌注特征与工作站适配)。

开发按“真实读取→桌面浏览→ROI/曲线→双灌注特征→开发队列→训练冻结→集成报告→独立验收”推进；每阶段配套测试与通过条件见[开发链与测试链](docs/DEMO_DESIGN.md#10-开发链与测试链)。临床资料与独立病例准备同时推进；当前两例用于开发，后续独立测试另行锁定。

D5从顶部“队列 → D5队列管理”进入，无需打开影像，已有窗口请先重启：

1. 点击“校验队列”：默认寻找旧清单与受控补录表。目前资料未准备，预期显示缺失列表、0行可用数据，冻结按钮不可用。
2. 点击“完全合成队列演示”：预期8次检查、7位患者、6行开发特征、1位独立保留患者。同一患者的第2次检查不单独入模；独立患者没有被导入开发矩阵的临床值、标签或特征。
3. 点击“重新校验并导出审计”保存逐例状态、缺失原因、特征及单独标签；点击“打开本次输出目录”查看。合成矩阵只测试工程流程，不能进入真实训练。
4. 合成示例可点击“重新校验并冻结患者划分”，验证带哈希的患者划分快照。真实项目需找回旧编号/清单，核对患者关联、开发用途、79/80对账及独立患者来源，才能冻结；冻结患者用途不代表临床资料/特征已合格。
5. 点击“导出空白补录表”，在`outputs/cohort_requests/<新目录>/`生成仅有表头的补录CSV与`cohort_project.json`，没有生成真实清单或编号。补齐后使用“打开队列配置”选择该配置再校验。临床/诊断CSV中的`patient_id`沿用原研究编号；同人多检查另用已审核`patient_key`关联。字段及逐步T5验收见[设计第16节](docs/DEMO_DESIGN.md#16-d5队列整理与患者划分)。

D5首版采用年龄、高血压、糖尿病、吸烟、静息心率、LDL-C六项候选临床字段，加PET MFR与MR MPRI；医院未确认字段/单位前不作为合格真实输入。未知值留空，按完整病例排除并记录原因，不填补为0。D4快照须完整、哈希匹配、关联/测量已复核且重算通过；缺PET/MR或QC未通过会排除。划分和交叉验证均以患者为单位，不随机拆切片，不训练模型。

Python及安装缓存统一放在项目外，避免复制/备份代码时携带整个运行环境：

```text
E:\Conda\
  miniconda3\       Conda管理器，base不安装项目依赖
  envs\cmvd\       CMVD专用Python 3.12.15与依赖
  cache\downloads\ 官方安装包
  cache\wheels\    已核验的依赖包，支持离线重建
```

项目保留`src/`源码、`scripts/`入口、`configs/`合同、`requirements/`依赖、`examples/`合成样例/空模板、`tests/`和`docs/`；`outputs/`只保留本地生成结果与原有证据，不进Git。表中的数据目录是既有约定，资料完整性另行核对，不因目录整理新建或复制病例清单。

直接依赖为numpy 2.5.4、pandas 3.0.6、scikit-learn 1.9.1、pydicom 3.0.1、PySide6 6.10.3、pyqtgraph 0.14.0；共18项完整依赖及wheel哈希见`requirements/windows-py312.lock`。重建使用[官方Miniconda安装包](https://repo.anaconda.com/miniconda/)，来源及SHA-256见`configs/runtime.json`。同版本环境在新位置重建，不直接搬动安装目录，不注册全局PATH。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup\bootstrap.ps1
```

该命令只为此进程设置脚本执行策略，不改变系统策略；安装管理器、创建专用环境并核验依赖后，生成被Git忽略的`configs/runtime.local.json`。默认根目录为`E:\Conda`，其他机器可传`-RuntimeRoot 'F:\Conda'`；已有完整环境复用，中断目录保留并拒绝自动覆盖。已备份完整缓存时可加`-Offline`离线重建。

合成框架本身仅使用Python标准库和人工固定系数，**不是训练好的模型、真实病例诊断或影像端到端演示**。以下测试与合成演示已在新环境通过：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\python.ps1 scripts/setup/check_environment.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\python.ps1 -m unittest discover -s tests -v
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\python.ps1 scripts/demo/run_demo.py --synthetic-demo
```

命令从脚本位置解析资源，结果写入`outputs/demo/`下独立新目录，不覆盖既有结果。真实训练尚未实现；D1–D4不会调用模拟诊断。历史病例清单与影像完整性尚未核对，具体诊断标准、临床资料、合格定量参考和独立测试来源仍待落实。

D1读取入口为`cmvd_demo.io`的`scan_case`、`build_grid`、`build_cine_stack`和`read_pixels`。数组轴为`[时间/相位序号, 层面, 行, 列]`；物理坐标为DICOM LPS，cine保留各层实际触发时间。接口、取消和已知边界见[设计稿第11节](docs/DEMO_DESIGN.md#11-d1读取接口)。只针对当前两个已解压开发样本的回归命令如下，不打开79个压缩包、不修改原件：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\python.ps1 scripts/data_inspection/validate_reader.py --samples
```

报告写入`outputs/reader_validation/`下独立新目录；包含受控研究元数据，不上传。脚本要求每组恰有一个解压样本，否则拒绝自动选取；当前样本协议/计数的回归断言仅用于此脚本，不是读取库的通用限制。其他MR序列和截图可索引，未纳入D1主线像素回归。

D2回归在原生Windows Qt窗口上执行；默认仅合成体模，加`--samples`才索引当前两例并核对主线首末帧。只保存合成窗口截图；报告在`outputs/desktop_validation/`，不生成患者截图或修改原件。自动交互已验证，T2独立查看器方向对照及T3人工操作验收待完成。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\python.ps1 scripts/desktop/check_desktop.py --samples
```

额外检查播放与连续快拖可加`--navigation`：对每例PET动态、MR灌注和短轴cine各检查一段播放及51次快速目标变更，报告Qt绘制帧率/间隔、空白观察和最终像素匹配。指标仅代表这两例的本机短程检查，不保证全队列或其他设备帧率。单帧预读最多4帧，共用64 MiB缓存，不加载整套4D影像。

D3回归加`--roi`；默认合成体模，加`--samples`后覆盖两例的PET静息/负荷动态、8个MR灌注平面与短轴cine。测试轮廓显式标为`engineering_fixture`，仅核对数值、逐帧修改及保存/重载，不能当作心肌标注或训练特征。快照及唯一合成截图保留在本次受控报告目录，不写原始影像。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\python.ps1 scripts/desktop/check_desktop.py --samples --roi --navigation
```

D4原生合成检查如下；可选`--roi-report <已完成D3报告>`复用其中的工程曲线核对派生计算/拒绝路径及空PET模板，不重新扫描源影像。报告写入`outputs/perfusion_validation/`，仅截图合成窗口。工程回归通过不代表每条真实曲线合格或研究验收通过。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\python.ps1 scripts/desktop/check_perfusion.py
```

查看影像可用3D Slicer导入单病例目录，如 `deidentified/阳性/POS-001`。

历史数据迁入并确认路径后，可用本地Conda执行全量头字段复核（历史约25分钟，每10,000文件输出进度，完成后更新audit中的报告；本次环境准备不执行此扫描）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\python.ps1 scripts/data_preparation/validate_deidentification.py
```

路径以脚本位置确定，不依赖启动目录。去标识脚本对现有输出设有拒绝覆盖保护；重建需先制定独立输出方案。

按实际功能扩展目录，不建立空目录或重复病例表。不因本次框架开发重跑既有影像处理。
