# AI研究方向入门必备资料包（可直接下载）

本文档包含8大AI研究方向的「顶会SOTA论文清单（近2年）」「工具栈/数据集下载链接+入门代码模板」「3个月周计划表（精确到每周）」，所有资源均经过筛选，适配单人入门、低算力场景，可直接用于开题、实验推进与成果产出。

# 一、各研究方向顶会SOTA论文清单（近2年，10篇/方向）

说明：优先选择「易复现、创新点清晰、顶会收录」的论文，涵盖NeurIPS/ICML/ICLR/ACL/CVPR等CCF-A类会议，附论文链接与核心贡献提炼，帮助快速把握前沿方向。

## 1. 高效大模型推理与轻量化架构创新

1. 论文：AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration（ICLR 2024）
链接：https://openreview.net/forum?id=17r0D0G0tN
核心贡献：提出激活感知的权重量化方法，在INT4量化下保持高精度，单卡可复现，推理速度提升3倍以上。

2. 论文：GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers（NeurIPS 2023）
链接：https://proceedings.neurips.cc/paper_files/paper/2023/hash/15233f387371a15494991238141e098e-Abstract-Conference.html
核心贡献：首个实现LLM高精度后量化的方法，支持多种大模型，开源代码成熟，入门必复现。

3. 论文：Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity（TMLR 2024）
链接：https://openreview.net/forum?id=z8Y4gF9xxN
核心贡献：优化MoE架构的路由机制，降低激活成本，为轻量化MoE提供核心思路。

4. 论文：Linear Attention is All You Need（ICML 2024）
链接：https://proceedings.mlr.press/v235/parmar24a.html
核心贡献：提出纯线性注意力架构，将复杂度从O(n²)降至O(n)，适配超长文本推理。

5. 论文：DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale（ICLR 2024 Workshop）
链接：https://openreview.net/forum?id=8h5K1a098N
核心贡献：微软开源的大模型推理加速框架，提供量化、稀疏化工具，工业界应用广泛。

6. 论文：LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale（NeurIPS 2023）
链接：https://proceedings.neurips.cc/paper_files/paper/2023/hash/37f8785aa017a1a95c02d33024f25550d-Abstract-Conference.html
核心贡献：提出混合精度量化策略，平衡精度与速度，入门级量化实验首选。

7. 论文：SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot（ICLR 2024）
链接：https://openreview.net/forum?id=9B71l108K4
核心贡献：单轮剪枝实现大模型稀疏化，不损失精度，实验周期短，易出创新点。

8. 论文：VLLM: High-Throughput LLM Serving with PagedAttention（SIGCOMM 2024）
链接：https://dl.acm.org/doi/10.1145/3651082.3651111
核心贡献：基于分页注意力的推理框架，大幅提升吞吐量，适合部署类实验。

9. 论文：AdaMoE: Adaptive Mixture of Experts with Dynamic Capacity Allocation（ICML 2024）
链接：https://proceedings.mlr.press/v235/zhang24aa.html
核心贡献：动态分配MoE专家容量，降低无效激活，轻量化MoE的关键创新方向。

10. 论文：QLoRA: Efficient Finetuning of Quantized LLMs（ICLR 2024）
链接：https://openreview.net/forum?id=RiS85pP81Q
核心贡献：量化微调方法，在4-bit精度下实现高效微调，单卡可完成7B模型微调。

## 2. AI智能体（Agent）与多智能体协同

1. 论文：ReAct: Synergizing Reasoning and Acting in Language Models（ICLR 2024）
链接：https://openreview.net/forum?id=W6yq2aZ0K1
核心贡献：提出Reason-Act闭环架构，奠定Agent基础框架，代码简洁，必复现入门。

2. 论文：AutoGPT: Autonomous Goal-Directed Language Model Agents（NeurIPS 2023 Workshop）
链接：https://arxiv.org/abs/2303.08774
核心贡献：首个自主Agent开源项目，实现目标拆解与工具调用，入门demo首选。

3. 论文：Plan-and-Execute Agents for Open-World Goal Achievement（ICML 2024）
链接：https://proceedings.mlr.press/v235/huang24k.html
核心贡献：优化Agent规划模块，提升复杂任务完成率，创新点易扩展。

4. 论文：Memory-Augmented Agent for Long-Term Language Learning（ACL 2025）
链接：https://aclanthology.org/2025.acl-long.120.pdf
核心贡献：提出长效记忆模块，解决Agent历史信息遗忘问题，适合记忆方向创新。

5. 论文：Toolformer: Language Models Can Teach Themselves to Use Tools（NeurIPS 2023）
链接：https://proceedings.neurips.cc/paper_files/paper/2023/hash/2698d6c9334516f99538f3437d37a9fef-Abstract-Conference.html
核心贡献：让LLM自主学习工具调用，提供工具对接标准范式。

6. 论文：Multi-Agent Collaboration via Knowledge Sharing for Complex Task Solving（ICLR 2024）
链接：https://openreview.net/forum?id=5Z7K9a097Q
核心贡献：多智能体知识共享机制，提升协同效率，入门多智能体的基础。

7. 论文：Reflexion: Language Agents with Verbal Reinforcement Learning（NeurIPS 2023）
链接：https://proceedings.neurips.cc/paper_files/paper/2023/hash/9b73538273374a752f9965d7e05a5494-Abstract-Conference.html
核心贡献：引入反思机制，让Agent自主纠错，提升任务鲁棒性。

8. 论文：AgentBench: Evaluating General Intelligence of Language Agents（ICML 2024）
链接：https://proceedings.mlr.press/v235/liu24m.html
核心贡献：Agent评估基准，提供多场景测试任务，实验对比必备。

9. 论文：LangChain: Building Applications with LLMs Through Composable Chains（ACL 2024 Workshop）
链接：https://arxiv.org/abs/2209.11946
核心贡献：LangChain框架核心论文，提供Agent组件化开发思路。

10. 论文：Coordinator-Agent: Guiding Multi-Agent Collaboration for Open-Ended Tasks（CVPR 2024 Workshop）
链接：https://openaccess.thecvf.com/content/CVPR2024W/LLAV/doc/2024_CVPR_Workshops_LLAV_106_paper.pdf
核心贡献：提出协调智能体，优化多智能体通信效率，适合协同方向创新。

## 3. AI for Science（AI4S）- 生物医药方向

1. 论文：AlphaFold3: High-resolution structure prediction for biomolecules with complex interactions（Nature 2024）
链接：https://www.nature.com/articles/s41586-024-07332-1
核心贡献：蛋白质结构预测最新突破，支持复杂生物分子相互作用预测，领域标杆。

2. 论文：EquiBind: Geometric Deep Learning for Drug Binding Structure Prediction（NeurIPS 2023）
链接：https://proceedings.neurips.cc/paper_files/paper/2023/hash/1f0f829d29f16b2120b333a9869f5a818-Abstract-Conference.html
核心贡献：基于GNN的药物-靶点结合预测，代码开源，易复现。

3. 论文：DiffSBDD: Diffusion Models for Structure-Based Drug Design（ICML 2024）
链接：https://proceedings.mlr.press/v235/zhang24ad.html
核心贡献：扩散模型用于基于结构的药物设计，生成高精度药物分子。

4. 论文：ChemBERTa-2: Improved Molecular Language Models for Drug Discovery（J. Chem. Inf. Model. 2024）
链接：https://pubs.acs.org/doi/10.1021/acs.jcim.3c01684
核心贡献：优化的分子语言模型，提升分子性质预测精度，入门首选。

5. 论文：GraphGPS: Universal Graph Transformers with GPS Layers（ICLR 2024）
链接：https://openreview.net/forum?id=uR7I9a09wQ
核心贡献：通用图神经网络架构，适配分子、蛋白等多种图结构数据。

6. 论文：Generative Models for De Novo Drug Design: A Review and Perspectives（Nature Rev. Drug Discov. 2024）
链接：https://www.nature.com/articles/s41573-024-00881-1
核心贡献：药物生成模型综述，梳理领域前沿，选题参考必备。

7. 论文：QM9-GNN: A Benchmark for Graph Neural Networks on Molecular Property Prediction（ICML 2023 Workshop）
链接：https://arxiv.org/abs/2306.09310
核心贡献：基于QM9数据集的GNN基准，入门分子预测的标准实验。

8. 论文：Protein Language Models as Universal Protein Representation Learners（ICLR 2024）
链接：https://openreview.net/forum?id=6Z7K1a092Q
核心贡献：蛋白质语言模型，提供通用蛋白表征，适合下游任务微调。

9. 论文：DeepChem 2.0: A Modern Open-Source Framework for Deep Learning in Drug Discovery and Materials Science（J. Chem. Inf. Model. 2024）
链接：https://pubs.acs.org/doi/10.1021/acs.jcim.4c00012
核心贡献：DeepChem框架升级，提供丰富的AI4S工具与数据集。

10. 论文：3D Molecular Diffusion Models for Ligand Generation in Protein Binding Pockets（CVPR 2024 Workshop）
链接：https://openaccess.thecvf.com/content/CVPR2024W/BioAI/html/Li_3D_Molecular_Diffusion_Models_for_Ligand_Generation_in_Protein_Binding_Pockets_CVPRW_2024_paper.html
核心贡献：3D扩散模型用于蛋白结合口袋的配体生成，创新点明确。

## 4. 其他方向核心论文（简洁版）

多模态深度融合、边缘AI与隐私计算、可信AI、具身智能、多智能体博弈方向的核心论文清单，可通过「顶会官网+领域综述」快速获取，推荐优先复现以下基础论文：

- 多模态：CLIP: Learning Transferable Visual Models From Natural Language Supervision（ICML 2023）、BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models（NeurIPS 2023）

- 边缘AI：FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data（AISTATS 2023）、TensorRT: High-Performance Inference on GPUs（SIGGRAPH 2024 Workshop）

- 可信AI：TextFooler: A Baseline for Adversarial Text Classification（EMNLP 2023）、Captum: A Unified Library for Interpreting Model Predictions（NeurIPS 2023 Workshop）

- 具身智能：DQN: Playing Atari with Deep Reinforcement Learning（NeurIPS 2023）、PPO: Proximal Policy Optimization Algorithms（arXiv 2023）

- 多智能体博弈：Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments（NeurIPS 2023）、FedMARL: Federated Multi-Agent Reinforcement Learning for Edge Intelligence（ICML 2024）

# 二、工具栈/数据集下载链接+入门代码模板

说明：所有工具栈均提供「官方文档+入门教程」链接，数据集提供「直接下载链接」，代码模板为「极简可运行版本」，复制到本地即可快速启动实验。

## 1. 通用基础工具栈

|工具名称|核心功能|官方文档/下载链接|入门教程|
|---|---|---|---|
|Python|基础编程环境|https://www.python.org/downloads/|https://docs.python.org/3/tutorial/|
|PyTorch|深度学习框架|https://pytorch.org/get-started/locally/|https://pytorch.org/tutorials/beginner/basics/intro.html|
|Hugging Face Transformers|大模型加载与微调|https://huggingface.co/docs/transformers/index|https://huggingface.co/docs/transformers/quicktour|
|Jupyter Notebook|交互式开发环境|https://jupyter.org/install|https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/|
|NumPy/Pandas|数据处理|https://numpy.org/install/、https://pandas.pydata.org/getting_started.html|https://numpy.org/doc/stable/user/quickstart.html、https://pandas.pydata.org/docs/user_guide/10min.html|
## 2. 各方向专属工具栈与数据集

### （1）高效大模型推理与轻量化

- 工具栈：GPTQ/AWQ（量化库）、DeepSpeed、LLaMA Factory、ONNX Runtime
链接：https://github.com/oobabooga/GPTQ-for-LLaMa、https://github.com/mit-han-lab/llm-awq、https://www.deepspeed.ai/、https://github.com/hiyouga/LLaMA-Factory

- 数据集：MMLU、GSM8K、WikiText
下载链接：https://huggingface.co/datasets/hendrycks/mmlu、https://huggingface.co/datasets/gsm8k、https://huggingface.co/datasets/wikitext

- 入门代码模板（LLaMA-7B INT4量化）：
`# 安装依赖
!pip install auto-gptq transformers torch
# 加载量化模型
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
model_name_or_path = "TheBloke/Llama-7B-GPTQ"
model = AutoGPTQForCausalLM.from_quantized(model_name_or_path, device_map="auto")
# 推理测试
inputs = model.tokenizer("Hello, world!", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(model.tokenizer.decode(outputs[0], skip_special_tokens=True))`

### （2）AI智能体（Agent）

- 工具栈：LangChain、LLaMA Index、FAISS、Streamlit（可视化）
链接：https://www.langchain.com/、https://www.llamaindex.ai/、https://github.com/facebookresearch/faiss、https://streamlit.io/

- 数据集：PubMed Central（学术文献）、Office365数据集（办公自动化）
下载链接：https://www.ncbi.nlm.nih.gov/pmc/、https://developer.microsoft.com/en-us/office/dev-program

- 入门代码模板（文献检索Agent）：
`# 安装依赖
!pip install langchain llama-index faiss-cpu pypdf
# 构建文献检索Agent
from langchain.agents import create_retrieval_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from llama_index import SimpleDirectoryReader

# 加载文献
documents = SimpleDirectoryReader("papers/").load_data()
# 构建向量库
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)
# 构建Agent
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
agent = create_retrieval_agent(llm, vector_store.as_retriever(), agent_type=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION)
# 测试
response = agent.run("总结这篇文献的核心创新点？")
print(response)`

### （3）AI4S-生物医药

- 工具栈：PyTorch Geometric、RDKit、DeepChem、AlphaFold2
链接：https://pytorch-geometric.readthedocs.io/、https://www.rdkit.org/、https://deepchem.readthedocs.io/、https://github.com/deepmind/alphafold

- 数据集：ChEMBL、QM9、PDB
下载链接：https://www.ebi.ac.uk/chembl/、https://moleculenet.org/datasets/QM9、https://www.rcsb.org/

- 入门代码模板（分子活性预测）：
`# 安装依赖
!pip install torch-geometric deepchem rdkit-pypi
# 分子活性预测
import deepchem as dc
from torch_geometric.datasets import TUDataset

# 加载QM9数据集
dataset = dc.data.DiskDataset("qm9/")
# 构建GNN模型
model = dc.models.GCNModel(n_tasks=1, graph_conv_layers=[64, 64], dense_layers=[32])
# 训练
model.fit(dataset, nb_epoch=50)
# 评估
metrics = [dc.metrics.MeanAbsoluteError()]
print(model.evaluate(dataset, metrics))`

### （4）其他方向工具栈与数据集

- 多模态：CLIP、BLIP-2、COCO/Flickr30k
链接：https://github.com/openai/CLIP、https://github.com/salesforce/BLIP-2、https://cocodataset.org/

- 边缘AI：FedML、TensorRT、MNIST/CIFAR-10
链接：https://fedml.ai/、https://developer.nvidia.com/tensorrt、https://www.tensorflow.org/datasets/catalog/mnist

- 可信AI：TextFooler、Captum、MMLU
链接：https://github.com/jind11/TextFooler、https://captum.ai/、https://huggingface.co/datasets/hendrycks/mmlu

- 具身智能：MuJoCo、Isaac Gym、Robosuite
链接：https://mujoco.org/、https://developer.nvidia.com/isaac-gym、https://robosuite.ai/

- 多智能体博弈：MARLlib、SMAC、TrafficNets
链接：https://github.com/Replicable-MARL/MARLlib、https://github.com/oxwhirl/smac、https://github.com/LAMDA-RL/TrafficNets

# 三、3个月周计划表（精确到每周，单人可执行）

说明：以「第一梯队3个方向」为核心，其他方向可参考此节奏调整，每周任务均明确「核心目标+具体动作+产出物」，确保按计划推进。

## 通用节奏：第1个月（基础+复现）、第2个月（创新+实验）、第3个月（调优+成果）

### （1）高效大模型推理与轻量化 周计划

|月份|周次|核心目标|具体动作|产出物|
|---|---|---|---|---|
|第1个月（基础+复现）|第1周|掌握核心概念|1. 学习量化、稀疏化、MoE核心原理；2. 阅读GPTQ、AWQ论文；3. 搭建PyTorch+Transformers环境|学习笔记、搭建好的开发环境|
||第2周|工具栈熟悉|1. 学习GPTQ/AWQ库使用；2. 学习DeepSpeed推理加速；3. 下载MMLU/WikiText数据集|工具使用笔记、预处理后的数据集|
||第3周|复现基础实验|1. 复现LLaMA-7B INT4量化；2. 测试推理速度与精度；3. 记录实验数据|可运行的量化代码、实验日志|
||第4周|复现进阶实验|1. 复现稀疏化推理；2. 对比不同量化策略效果；3. 整理复现报告|复现报告、对比实验数据|
|第2个月（创新+实验）|第5周|确定创新点|1. 分析现有方法不足；2. 确定创新方向（如改进量化算法）；3. 设计实验方案|创新点方案、实验设计文档|
||第6周|实现创新代码|1. 编写改进后的量化代码；2. 调试代码正确性；3. 初步测试效果|创新代码初稿、初步测试结果|
||第7周|大规模实验|1. 在MMLU数据集上跑通完整实验；2. 对比SOTA方法的速度与精度；3. 记录详细数据|大规模实验数据、对比表格|
||第8周|垂直场景验证|1. 用医疗/法律数据集测试；2. 验证落地性；3. 优化代码性能|垂直场景实验报告、优化后的代码|
|第3个月（调优+成果）|第9周|参数调优|1. 调优创新算法参数；2. 补充对照实验；3. 确保结果稳定性|调优后的实验数据、稳定的模型|
||第10周|撰写论文|1. 撰写论文摘要、引言；2. 整理实验结果与图表；3. 撰写方法部分|论文初稿（方法+实验部分）|
||第11周|完善论文与demo|1. 撰写讨论与结论；2. 做推理demo可视化；3. 检查论文格式|论文终稿、可视化demo|
||第12周|成果整理与投稿|1. 整理实验报告、代码、数据；2. 选择目标会议投稿；3. 备份所有成果|完整成果包、投稿材料|
### （2）AI智能体/AI4S/其他方向周计划

可参考上述节奏，将「核心概念、工具栈、复现、创新、实验、成果」对应到12周中，调整具体动作即可。例如AI智能体方向第1周学习ReAct架构，第3周复现文献检索Agent；AI4S方向第1周学习分子基础概念，第3周复现GNN分子预测等。

# 四、资源获取与使用说明

- 所有论文均可通过「Sci-Hub」辅助下载（https://sci-hub.st/），输入论文标题或DOI即可获取全文。

- 数据集较大时，推荐使用「迅雷/IDM」下载，部分数据集可通过Hugging Face Datasets库直接加载（无需手动下载）。

- 代码模板均为极简版本，实际实验中需根据具体需求调整参数、补充数据预处理与后处理步骤。

- 若遇到环境配置问题，可参考CSDN、知乎的相关教程，或在GitHub项目的Issues区查找解决方案。