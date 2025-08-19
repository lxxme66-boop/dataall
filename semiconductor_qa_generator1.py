# -*- coding: utf-8 -*-
"""
半导体显示技术领域智能QA生成系统 - 核心业务逻辑模块
包含完整的三步评估流程：文本质量评估、问题生成、问题质量评估
"""
import os
import re
import json
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer
from tqdm import tqdm
import logging
from pathlib import Path
import time
from datetime import datetime
import torch 
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 尝试导入vLLM，如果失败则使用模拟类
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    logger.warning("vLLM not available, using mock implementation")
    VLLM_AVAILABLE = False
    
    class LLM:
        """Mock LLM class for when vLLM is not available"""
        def __init__(self, **kwargs):
            self.model_path = kwargs.get('model')
            logger.warning(f"Mock LLM initialized with model: {self.model_path}")
            
        def generate(self, prompts, sampling_params, use_tqdm=True):
            """Mock generate method"""
            results = []
            for prompt in prompts:
                # 模拟输出
                mock_output = type('Output', (), {
                    'outputs': [type('GeneratedOutput', (), {
                        'text': '【是】' if '评分' in prompt else '[[1]] 示例问题1\n[[2]] 示例问题2\n[[3]] 示例问题3'
                    })()]
                })()
                results.append(mock_output)
            return results
    
    class SamplingParams:
        """Mock SamplingParams class"""
        def __init__(self, **kwargs):
            self.temperature = kwargs.get('temperature', 0.6)
            self.repetition_penalty = kwargs.get('repetition_penalty', 1.1)


@dataclass
class ModelConfig:
    """模型配置类"""
    name: str
    path: str
    stop_tokens: List[Any]
    max_model_len: int = 96 * 1024
    gpu_memory_utilization: float = 0.95
    tensor_parallel_size: int = 4
    temperature: float = 0.6
    repetition_penalty: float = 1.1
    min_p: float = 0
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 4096


# 预定义的模型配置
MODEL_CONFIGS = {
    "qwen-vllm": ModelConfig(
        name="qwen-vllm",
        path="/mnt/data/MLLM/liuchi/trained_models/Qwen3-32B-dpo-5w_retrain",
        stop_tokens=[151329, 151336, 151338]
    ),
    "qw2_72": ModelConfig(
        name="qw2_72",
        path="/data/lc/openmodels/qw2_72b_instruct",
        stop_tokens=["<|im_end|>"]
    ),
    "qw2_72_awq": ModelConfig(
        name="qw2_72_awq",
        path="/data/lc/openmodels/qw2_72b_instruct_awq",
        stop_tokens=["<|im_end|>"]
    ),
    "qw2.5_32": ModelConfig(
        name="qw2.5_32",
        path="/data/lc/openmodels/qw2.5_32b_instruct",
        stop_tokens=["<|im_end|>"]
    ),
    "qw2.5_72": ModelConfig(
        name="qw2.5_72",
        path="/data/lc/openmodels/qw2.5_72b_instruct",
        stop_tokens=["<|im_end|>"]
    ),
    "llama3.1_70": ModelConfig(
        name="llama3.1_70",
        path="/data/lc/openmodels/llama3.1_70b_instruct",
        stop_tokens=["<|eot_id|>"]
    )
}


class SemiconductorQAGenerator:
    def __init__(self, batch_size=4, gpu_devices=None):
        """初始化QA生成器 - 修复model_name参数问题"""
        self.batch_size = batch_size
        self.gpu_devices = gpu_devices or "0"
        self.model_name = "qwen-vllm"  # 默认模型名称
        
        # 加载模型配置
        self.config = MODEL_CONFIGS.get(self.model_name)
        if not self.config:
            logger.warning(f"未找到模型 {self.model_name} 的配置，使用默认配置")
            self.config = ModelConfig(
                name=self.model_name,
                path=self.model_name,
                stop_tokens=[],
                max_model_len=96 * 1024
            )
        self.stats = {
            "generated_questions": 0,
            "total_questions": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
        }
        
        # 原有的初始化代码保持不变
        self.tokenizer = None
        self.model = None
        self.device_map = self._setup_device_map()
        
        # 加载模板
        self._load_templates()
        
        # 初始化vLLM相关
        self.llm = None
        self.sampling_params = None
        
        # 自动初始化LLM
        self.setup_llm_and_tokenizer()
        
        logger.info(f"初始化QA生成器完成: batch_size={batch_size}, gpu_devices={gpu_devices}")
    
    def setup_llm_and_tokenizer(self):
        """初始化vLLM和tokenizer"""
        if self.llm is not None:
            logger.info("LLM已经初始化，跳过")
            return
            
        logger.info(f"初始化vLLM模型: {self.config.path}")
        
        try:
            # 设置环境变量
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
            
            # 初始化vLLM
            if VLLM_AVAILABLE:
                self.llm = LLM(
                    model=self.config.path,
                    trust_remote_code=True,
                    gpu_memory_utilization=0.95,
                    tensor_parallel_size=len(self.gpu_devices.split(',')) if ',' in self.gpu_devices else 1,
                    max_model_len=self.config.max_model_len
                )
                
                # 初始化tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.path, trust_remote_code=True)
                
                # 设置sampling参数
                self.sampling_params = SamplingParams(
                    temperature=0.6,
                    repetition_penalty=1.1,
                    max_tokens=8192,
                    stop=self.config.stop_tokens if self.config.stop_tokens else None
                )
                
                logger.info("vLLM和tokenizer初始化成功")
            else:
                logger.warning("vLLM不可用，使用模拟实现")
                self.llm = LLM(model=self.config.path)
                self.tokenizer = None  # Mock模式下不需要真实tokenizer
                self.sampling_params = SamplingParams(
                    temperature=0.6,
                    repetition_penalty=1.1,
                    max_tokens=8192
                )
                
        except Exception as e:
            logger.error(f"初始化vLLM失败: {str(e)}")
            import traceback
            logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
            raise RuntimeError(f"无法初始化vLLM模型 {self.config.path}") from e
    
    def _format_messages(self, messages, add_generation_prompt=True, truncation=True):
        """格式化消息，处理tokenizer可能为None的情况"""
        if self.tokenizer is not None:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                truncation=truncation
            )
        else:
            # Mock模式或tokenizer未初始化时，直接构建提示
            formatted = ""
            for msg in messages:
                role = msg['role'].capitalize()
                content = msg['content']
                formatted += f"{role}: {content}\n\n"
            if add_generation_prompt:
                formatted += "Assistant:"
            return formatted
    
    def _encode_length(self, text):
        """获取文本编码长度，处理tokenizer可能为None的情况"""
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        else:
            # 粗略估计：每个字符算1个token
            return len(text)

    def set_model_name(self, model_name: str):
        """设置模型名称 - 新增方法"""
        self.model_name = model_name
        logger.info(f"设置模型名称为: {model_name}")
        
        # 更新模型配置
        self.config = MODEL_CONFIGS.get(self.model_name)
        if not self.config:
            logger.warning(f"未找到模型 {self.model_name} 的配置，使用默认配置")
            self.config = ModelConfig(
                name=self.model_name,
                path=self.model_name,
                stop_tokens=[],
                max_model_len=96 * 1024
            )
        # 如果模型已经加载，重新加载新模型
        if self.model:
            logger.info("模型已加载，重新加载新模型...")
            self._load_model()
            
        # 重新初始化vLLM
        if self.llm:
            logger.info("重新初始化vLLM...")
            self.llm = None
            self.tokenizer = None
            self.sampling_params = None
            self.setup_llm_and_tokenizer()
    
    def load_model(self):
        """加载模型（公有方法）"""
        if self.model is None:
            self._load_model()
        else:
            logger.info("模型已加载，跳过加载")
    
    def _load_model(self):
        """加载模型和分词器"""
        logger.info(f"加载模型: {self.model_name}")
        
        try:
            # 实际加载模型的代码
            # 这里应该是加载tokenizer和模型的代码
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=torch.float16
            )
            logger.info(f"成功加载模型: {self.model_name}")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            # 添加详细错误信息
            import traceback
            logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
            raise RuntimeError(f"无法加载模型 {self.model_name}") from e
    
    def _setup_device_map(self):
        """设置GPU设备映射"""
        if not torch.cuda.is_available():
            logger.warning("未检测到GPU，将使用CPU")
            return {"": "cpu"}
        
        # 处理可能的空值或无效值
        if not self.gpu_devices or not isinstance(self.gpu_devices, str):
            logger.warning(f"无效的GPU设备配置: {self.gpu_devices}，使用默认GPU 0")
            return {"": "cuda:0"}
        
        try:
            device_ids = [int(id) for id in self.gpu_devices.split(",") if id.isdigit()]
            if not device_ids:
                logger.warning("未指定有效的GPU设备，使用默认GPU 0")
                return {"": "cuda:0"}
            
            # 创建设备映射
            device_map = {f"cuda:{i}": f"cuda:{id}" for i, id in enumerate(device_ids)}
            logger.info(f"创建设备映射: {device_map}")
            return device_map
        except Exception as e:
            logger.error(f"创建设备映射失败: {str(e)}，使用默认GPU 0")
            return {"": "cuda:0"}
    
    # 其他方法保持不变...
    
    def _load_templates(self):
        """加载评估模板"""
        # 文本质量评分模板
        self.score_template = """你的任务是依据以下评分规则对文本质量进行打分，并输出最终得分。评分流程如下：
1.依照每个标准依次评估文本。对每个子问题如实作答。若对某子问题答案为明确 "是"，则按标准相应加分或减分;
2.记录每个标准的累计分数，得出总分;
3.依据以下说明，将最终评估结果整理为有效的 JSON 对象。

## 评分标准：
1.标准 1：问题完整性
(1) 内容无清晰主要问题，或缺乏足够线索得出正确答案，得 0 分。
(2) 内容包含一个主要问题，且有足够线索得出正确答案，得 + 1 分。
(3) 文本体现多位作者间互动与讨论，如提出答案、评估反思答案、回应批评、修订编辑答案，得 + 1 分。
2.标准 2：问题复杂性和技术深度
(1) 内容难度为大学水平或以下，得 0 分。
(2) 内容难度为研究生水平或以上，仅领域专家能理解，得 + 1 分。
(3) 所讨论问题极具挑战性，高技能非专家花费 30 分钟上网搜索或阅读文献后，仍无法完全理解问题或给出正确答案，得 + 1 分。
3.标准 3：技术正确性和准确性
(1) 文本含显著技术错误或不准确，得 -1 分。
(2) 文本有一定技术正确性，但存在明显缺陷或遗漏（如单位错误、推导不完整），得 0 分。
(3) 文本技术正确，但有小缺陷或遗漏（如小代数错误、解释不完整），得 + 0.5 分。
(4) 文本技术高度正确，解释清晰准确（如精确定义、完整推导），得 + 0.5 分。
(5) 文本技术卓越正确，解释严格精确（如形式化证明、精确计算），得 + 1 分。
4.标准 4：思维和推理
(1) 文本无任何思维或推理迹象，得 -1 分。
(2) 文本展现一些基本思维和推理能力（如直接应用已知技术、简单分析问题），得 + 0.5 分。
(3) 文本展现一定思维和推理能力（如考虑多种解决方法、讨论不同方案权衡），得 + 0.5 分。
(4) 文本展现显著思维和推理能力（如通过多步推理链解决复杂问题、运用专业科学领域高级推理模式），得 + 1 分。
(5) 文本展现卓越思维和推理能力（如以高度创新方式解决专业领域复杂问题、结合多种推理技术对问题进行新抽象），得 + 1 分。

最终评判标准：若各项标准得分均大于零，且标准 4 得分大于等于 1 分，则该文本内容适合生成逻辑推理问题。

[文本内容的开始]
{academic_paper}
[文本内容的结束]

格式要求：只输出文本内容是否适合生成复杂推理问题，不输出任何别的内容。并且是否适合严格按照以下格式进行输出：
【是】或者【否】。不要输出为空，不要输出其他内容，输出是或否时，要带上【】符号进行输出。
"""

        # 问题生成模板 - 使用用户提供的专业版本
        self.prompt_template = """你是一位半导体显示技术领域的资深专家，擅长从技术文献中提炼核心知识点。你的职责是从论文中生成问题和相应的答案，问题和相应的答案对需要提供给资深的人员学习，问题和相应的答案的质量要高。请根据输入的学术论文内容，生成3个需要逻辑推理才能解答的高质量技术问题，请确保这些问题能够直接从论文中找到答案。这些问题将用于资深研究人员的专业能力评估，需满足以下要求：

<think>
首先，我需要仔细阅读这篇学术论文，理解其核心内容和技术要点。

让我分析论文的主要内容：
1. 识别论文讨论的核心技术问题
2. 找出涉及逻辑推理的技术原理和机制
3. 确定可以设计深度问题的关键点

接下来，我将基于以下原则设计问题：
- 问题必须基于论文内容，答案可以从论文中找到
- 问题需要逻辑推理才能解答，不能是简单的事实性问题
- 问题描述要清晰、完整、专业
- 避免使用"本文"、"论文"等自指表述
- 确保问题具有技术深度和挑战性
</think>

##【核心要求】
### 问题设计准则：
(1) 仔细通读全文，找出涉及逻辑推理的文本部分，据此设计相关问题。
(2) 问题要基于论文里的技术原理，描述务必清晰、全面、明确。问题中主语或名词的表述要精准、全面且通用。
(3) 避免在问题中引用文献或文章自定义的专有名词。利用自身半导体显示领域知识，生成通用问题，确保不读论文也能理解问题含义。
(4) 问题中的名词描述必须和论文一致，不能缩写。比如论文是 "OLED 材料"，问题不能写成 "材料"；论文是 "LTPS 器件"，问题不能写成 "器件"。
(5) 提问不能针对论文里的某个特定示例。要让顶尖科学家不读论文也能理解并回答问题。问题里不能有 "书本""论文""本文""本实验" 等类似表述。
(6) 确保生成问题完整，与论文彻底解耦，不依赖论文内容。若问题有背景信息，必须阐述清楚。
(7) 问题简洁：生成的问题要凝练、简洁。

### 问题设计的科学严谨性：
(1) 因果链：问题需呈现完整技术逻辑链，比如 "机制 A 怎样影响参数 B，进而引发现象 C"。
(2) 周密性：思考过程要科学严谨，逐步推进。保证问题及对应答案源自论文内容，且答案在论文中有详细阐述。

## 【禁止事项】
× 禁止使用"本文/本研究/本实验"等论文自指表述
× 禁止提问孤立概念（如：XX技术的定义是什么）
× 禁止超出论文技术范围的假设性问题

##【格式要求】
用中文输出。当前阶段只设计问题，不输出答案。输出问题前必须用 </think> 结束思考后在输出问题。严格按照以下格式输出你设计的问题：
[[1]] 第1个问题
[[2]] 第2个问题
[[3]] 第3个问题

[学术论文的开始]
{academic_paper}
[学术论文的结束]
"""

        # 问题质量评估模板
        self.evaluator_template = """您是一位专家评估员，负责决定问题是否符合推理问题标准。您的评估必须结合给定文章内容和给定问题判断。

## 【评估标准】
### 因果性：
(1) 问题应展现出完整的技术逻辑链。比如，类似 "机制 A 怎样影响参数 B，最终致使现象 C 出现" 这种形式。

### 周密性：
(1) 思维过程要科学且严谨，需逐步思考。问题及对应的答案必须源于论文内容，且答案在论文中要有详细描述。

### 完整性：
(1) 问题是否全面涵盖文章相关内容的各个方面？
(2) 问题描述应简洁凝练，语义完整。
(3) 问题要与文章内容完全独立，不依赖文章也能被清晰理解，即问题需完整、自足。

[文章内容的开始]
{academic_paper}
[文章内容的结束]

[问题内容]
{academic_question}

格式要求：仅输出文本内容生成的问题是否符合标准，严格按以下格式，有且仅输出【是】或者【否】，不输出任何别的内容，不能输出为空，输出是或否时，要带上【】符号进行输出。用中文输出，严格按照以下格式进行输出：【是】或者【否】
"""

    def is_to_drop(self, text: str) -> bool:
        """
        判断文本是否需要被过滤掉
        
        Args:
            text: 待检查的文本
            
        Returns:
            bool: True表示需要过滤，False表示保留
        """
        text = text.strip()[:10]    
        patterns = ["", "#"]
        for pattern in patterns:
            if text == pattern:
                return True 
                
        patterns = [
            r'http://www.cnki.net', r'https://www.cnki.net', r'^\[\d{1,4}\]', 
            r'^\*\s+\[\d{1,4}\]', r'^\*\s+\(\d{1,4}\)', 
            r'^致谢.*[0-9]$', r'.*致\s*谢.*', r'.*目\s*录.*', 
            r'\.\.\.\.\.\.\.\.', r'\…\…\…', r"(http://www|doi:|DOI:|please contact)",
            r"(work was supported by|study was supported by|China|Republic of Korea|Authorized licensed use limited to)",
            r"\s[1-9]\d{5}(?!\d)",  # 邮编
            r"\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*", 
            r"(received in revised form|All rights reserved|©)", r"[a-zA-z]+://[^\s]*",
            r"(13[0-9]|14[5|7]|15[0|1|2|3|5|6|7|8|9]|18[0|1|2|3|5|6|7|8|9])\d{8}", 
            r"\d{3}-\d{8}|\d{4}-\d{7}",
            r'^分\s*类\s*号', r'^学\s*科\s*专\s*业', r'^签\s*字\s*日\s*期', 
            r'^申\s*请\s*人\s*员\s*姓\s*名',
            r'^日\s*期', r'^指\s*定\s*教\s*师', r'学\s*位\s*论\s*文', 
            r'^工\s*作\s*单\s*位', r'^电\s*话', r'^通讯地址', r'^邮\s*编',
            r'^\[?\d+\]?', r'^\s*\[?\d+\]?', r'^\［?\d+\］?', r'^\s*\［?\d+\］?'
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        keyword_patterns = [
            '申请号|专利号|已录用|学报|研究生|已收录|攻读|第一作者|第二作者|参考文献|专业名称|863项目|导师',
            '教授|感谢|致谢|谢谢|指导|朋友|家人|亲友|师弟|师妹|老师|同学|父母|充实|答辩|祝愿|独创性声明|作者签名',
            '发表文章|论文使用授权声明|本人|知网|论文使用权|发表的论文|申请的专利|申请专利|发表的文章|发表学术论文|发表论文',
            '参与科研项目|作者简介|三年的学习|大学硕士学位论文|大学博士学位论文|涉密论文|学校代码|论文提交日期|委员：|中图分类号',
            '原创性声明|顺利完成学业|All rights reserved|参 考 文 献|参考文献|所在学院|国家自然科学基金|教育部重点学科建设',
            '时间飞梭|时光飞梭|光阴似箭|白驹过隙|论文版权|本学位论文|使用授权书|References|Acknowledgements',
            '论文著作权|保密的学位论文|中国第一所现代大学|参加科研情况|独 创 性 声 明|论文使用授权|获得的专利|家庭的爱|文献标识码|文章编号'
        ]
        
        for pattern in keyword_patterns:
            if re.findall(pattern, text):
                return True   
        
        # 判断是否不包含中文字符
        num = 0
        for t in text:
            if '\u4e00' <= t <= '\u9fa5':
                num += 1    
        if len(text) > 0 and num / len(text) < 0.01:
            return True
                    
        return False

    def drop(self, texts: str, concatenation: str = "\n") -> str:
        """
        过滤文本中的无关内容
        
        Args:
            texts: 输入文本
            concatenation: 连接符
            
        Returns:
            str: 过滤后的文本
        """
        new_texts = []
        texts = texts.split("\n")
        for i, text in enumerate(texts):
            if not self.is_to_drop(text):
                new_texts.append(text)
        return concatenation.join(new_texts)

    def load_paper(self, file_path: str) -> str:
        """
        加载论文并进行过滤处理
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 过滤后的论文内容
        """
        with open(file_path, "r", encoding="utf8") as f:
            content = f.read()
        deal_content = self.drop(content)
        return deal_content

    def to_batch(self, lst: List[Any], groupsize: int) -> List[List[Any]]:
        """
        将列表分批处理
        
        Args:
            lst: 输入列表
            groupsize: 每批大小
            
        Returns:
            list: 分批后的列表
        """
        return [lst[i:i+groupsize] for i in range(0, len(lst), groupsize)]

    async def judge_md_data(self, raw_folders: List[str], save_paths: List[str], 
                           jsonl_file_path: str) -> Dict[str, Any]:
        """
        评估文档数据质量，判断是否适合生成逻辑推理问题
        
        Args:
            raw_folders: 原始文件夹列表
            save_paths: 保存路径列表
            jsonl_file_path: JSONL输出文件路径
            
        Returns:
            dict: 评估结果统计
        """
        start_time = time.time()
        total_stats = {
            "total_files": 0,
            "passed_files": 0,
            "failed_files": 0,
            "error_files": 0
        }
        
        for raw_folder, save_path in zip(raw_folders, save_paths):
            logger.info(f"Processing folder: {raw_folder}")
            
            # 获取所有txt文件
            files = []
            if os.path.exists(raw_folder):
                files = [f for f in os.listdir(raw_folder) if f.endswith('.txt')]
            else:
                logger.error(f"Folder not found: {raw_folder}")
                continue
                
            files.sort()
            total_stats["total_files"] += len(files)
            
            # 检查已处理的文件
            already_ids = []
            results = []
            if os.path.exists(save_path):
                try:
                    with open(save_path, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    already_ids = [r["id"] for r in results if "id" in r]
                except Exception as e:
                    logger.error(f"Error loading existing results: {e}")
            
            logger.info(f"Already processed: {len(already_ids)} files")
            
            # 待处理文件
            to_do = [f for f in files if f not in already_ids]
            if not to_do:
                logger.info("No new files to process")
                continue
                
            # 批处理
            batches = self.to_batch(to_do, self.batch_size)
            comply_stand_datas = []
            
            for batch_idx, batch in enumerate(tqdm(batches, desc=f"Judging {raw_folder}")):
                score_inputs = []
                batch_metadata = []
                
                # 准备批处理输入
                for paper in batch:
                    try:
                        paper_name = paper.split("_part")[0]
                        paper_content = self.load_paper(os.path.join(raw_folder, paper))
                        
                        if len(paper_content.strip()) < 100:
                            logger.warning(f"Paper {paper} too short, skipping")
                            continue
                        
                        # 构建评分提示
                        score_prompt = self.score_template.replace("{academic_paper}", paper_content)
                        score_messages = [
                            {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                            {"role": "user", "content": score_prompt}
                        ]
                        
                        # 应用聊天模板
                        data_li = self._format_messages(score_messages)
                        
                        # 检查长度并截断
                        #if len(self.tokenizer.encode(data_li)) > self.config.max_model_len - 1024:
                        # 检查长度并截断
                        if self._encode_length(data_li) > self.config.max_model_len - 1024:
                            logger.warning(f"Prompt too long for {paper}, truncating")
                            # 截断内容
                            max_content_len = self.config.max_model_len - 2048
                            truncated_content = paper_content[:max_content_len]
                            score_prompt = self.score_template.replace("{academic_paper}", truncated_content)
                            score_messages = [
                                {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                                {"role": "user", "content": score_prompt}
                            ]

                            data_li = self._format_messages(score_messages)
                        
                        
                        score_inputs.append(data_li)
                        batch_metadata.append({
                            "paper": paper,
                            "paper_name": paper_name,
                            "paper_content": paper_content
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing {paper}: {e}")
                        total_stats["error_files"] += 1
                        continue
                
                if not score_inputs:
                    continue
                
                # 批量推理
                try:
                    score_outputs = self.llm.generate(score_inputs, self.sampling_params, use_tqdm=False)
                    
                    # 处理输出
                    for idx, output in enumerate(score_outputs):
                        metadata = batch_metadata[idx]
                        score_text = output.outputs[0].text.strip()
                        
                        # 提取最后一行作为判断结果
                        lines = score_text.split('\n')
                        final_judgment = lines[-1] if lines else ""
                        
                        # 判断是否通过
                        is_suitable = '【是】' in final_judgment
                        
                        result = {
                            "stats": 1 if is_suitable else 0,
                            "paper_name": metadata["paper_name"],
                            "paper_content": metadata["paper_content"],
                            "score_text": score_text,
                            "judgment": final_judgment,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        comply_stand_datas.append(result)
                        
                        if is_suitable:
                            total_stats["passed_files"] += 1
                        else:
                            total_stats["failed_files"] += 1
                            
                except Exception as e:
                    logger.error(f"Error in batch inference: {e}")
                    total_stats["error_files"] += len(batch_metadata)
                
                # 定期保存结果
                if (batch_idx + 1) % self.save_steps == 0:
                    self._save_intermediate_results(comply_stand_datas, jsonl_file_path)
            
            # 保存最终结果
            if comply_stand_datas:
                self._save_intermediate_results(comply_stand_datas, jsonl_file_path)
                
                # 更新总结果文件
                all_results = results + [
                    {"id": d["paper_name"], "paper_content": d["paper_content"], 
                     "stats": d["stats"], "score_text": d["score_text"]}
                    for d in comply_stand_datas if d["stats"] == 1
                ]
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=4)
                
                logger.info(f"Results saved to {save_path}")
    
        # 统计处理时间
        total_stats["processing_time"] = time.time() - start_time
        self.stats.update(total_stats)
        
        logger.info(f"Text quality assessment completed: {total_stats}")
        return total_stats

    async def judge_processed_texts(self, processed_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        评估已处理的文本数据质量，判断是否适合生成逻辑推理问题
        
        Args:
            processed_texts: 已处理的文本列表，每个元素包含:
                - paper_name: 文档名称
                - md_content: Markdown格式的内容
                - source_info: 原始信息
                
        Returns:
            list: 评估结果列表，每个元素包含原始数据和判断结果
        """
        start_time = time.time()
        logger.info(f"Starting text quality evaluation for {len(processed_texts)} texts")
        
        # 创建批次
        batch_size = self.batch_size
        batches = [processed_texts[i:i + batch_size] for i in range(0, len(processed_texts), batch_size)]
        
        all_results = []
        total_passed = 0
        total_failed = 0
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating text quality")):
            score_inputs = []
            batch_metadata = []
            
            # 准备批处理输入
            for item in batch:
                try:
                    paper_content = item.get('md_content', '')
                    
                    if len(paper_content.strip()) < 100:
                        logger.warning(f"Text {item.get('paper_name', 'unknown')} too short, marking as failed")
                        all_results.append({
                            **item,
                            'judgment': {
                                'suitable_for_qa': False,
                                'reason': '文本内容过短',
                                'score_text': '【否】文本内容过短，不足以生成有意义的问题'
                            }
                        })
                        total_failed += 1
                        continue
                    
                    # 构建评分提示
                    score_prompt = self.score_template.replace("{academic_paper}", paper_content)
                    score_messages = [
                        {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                        {"role": "user", "content": score_prompt}
                    ]
                    
                    # 应用聊天模板
                    data_li = self._format_messages(score_messages)
                    
                    
                    # 检查长度并截断
                    # 检查长度并截断
                    if self._encode_length(data_li) > self.config.max_model_len - 1024:
                        logger.warning(f"Prompt too long for {item.get('paper_name', 'unknown')}, truncating")
                        # 截断内容
                        max_content_len = self.config.max_model_len - 2048
                        truncated_content = paper_content[:max_content_len]
                        score_prompt = self.score_template.replace("{academic_paper}", truncated_content)
                        score_messages = [
                            {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                            {"role": "user", "content": score_prompt}
                        ]
                        data_li = self._format_messages(score_messages)
                    
                    
                    score_inputs.append(data_li)
                    batch_metadata.append(item)
                    
                except Exception as e:
                    logger.error(f"Error processing text {item.get('paper_name', 'unknown')}: {e}")
                    all_results.append({
                        **item,
                        'judgment': {
                            'suitable_for_qa': False,
                            'reason': f'处理错误: {str(e)}',
                            'score_text': f'【否】处理过程中出现错误: {str(e)}'
                        }
                    })
                    total_failed += 1
                    continue
            
            # 批量推理
            if score_inputs:
                try:
                    score_outputs = self.llm.generate(
                        score_inputs,
                        self.sampling_params,
                        use_tqdm=False
                    )
                    
                    # 处理输出
                    for output, metadata in zip(score_outputs, batch_metadata):
                        score_text = output.outputs[0].text.strip()
                        
                        # 判断是否通过评估
                        suitable = "【是】" in score_text or "是】" in score_text
                        
                        result = {
                            **metadata,
                            'judgment': {
                                'suitable_for_qa': suitable,
                                'reason': '通过质量评估' if suitable else '未通过质量评估',
                                'score_text': score_text
                            }
                        }
                        
                        all_results.append(result)
                        
                        if suitable:
                            total_passed += 1
                        else:
                            total_failed += 1
                            
                except Exception as e:
                    logger.error(f"Error in batch inference: {e}")
                    # 批量推理失败，将整批标记为失败
                    for metadata in batch_metadata:
                        all_results.append({
                            **metadata,
                            'judgment': {
                                'suitable_for_qa': False,
                                'reason': f'批量推理错误: {str(e)}',
                                'score_text': f'【否】批量推理过程中出现错误: {str(e)}'
                            }
                        })
                        total_failed += 1
        
        # 记录统计信息
        processing_time = time.time() - start_time
        logger.info(f"Text quality evaluation completed in {processing_time:.2f}s")
        logger.info(f"Total: {len(processed_texts)}, Passed: {total_passed}, Failed: {total_failed}")
        
        return all_results

    def _save_intermediate_results(self, results: List[Dict], jsonl_path: str):
        """保存中间结果到JSONL文件"""
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def generate_question_data(self, jsonl_file_input: str, jsonl_file_output: str, add_prompt: str = "") -> Dict[str, Any]:
        """
        根据通过质量评估的文本生成问题
        
        Args:
            jsonl_file_input: 输入JSONL文件路径
            jsonl_file_output: 输出JSONL文件路径
            add_prompt: 额外的提示信息，用于指定问题类型和示例
            
        Returns:
            dict: 生成结果统计
        """
        start_time = time.time()
        stats = {
            "total_processed": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_questions": 0
        }
        
        # 读取输入数据
        suitable_papers = []
        with open(jsonl_file_input, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("stats") == 1:
                        suitable_papers.append(data)
                except Exception as e:
                    logger.error(f"Error parsing line: {e}")
                    continue
        
        logger.info(f"Found {len(suitable_papers)} suitable papers for question generation")
        stats["total_processed"] = len(suitable_papers)
        
        # 批处理生成问题
        batches = self.to_batch(suitable_papers, self.batch_size)
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Generating questions")):
            inputs = []
            batch_metadata = []
            
            # 准备输入
            for paper_data in batch:
                try:
                    paper_name = paper_data["paper_name"]
                    paper_content = paper_data["paper_content"]
                    
                    # 构建生成提示 - 如果有额外提示，则添加到模板中
                    if add_prompt:
                        # 在模板的要求部分后添加额外提示
                        enhanced_template = self.prompt_template.replace(
                            "### 输出格式要求：",
                            f"{add_prompt}\n\n### 输出格式要求："
                        )
                        generate_prompt = enhanced_template.replace("{academic_paper}", paper_content)
                    else:
                        generate_prompt = self.prompt_template.replace("{academic_paper}", paper_content)
                    
                    messages = [
                        {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                        {"role": "user", "content": generate_prompt}
                    ]
                    
                    # 应用聊天模板
                    prompt = self._format_messages(messages)
                    
                    # 检查长度
                    # 检查长度
                    if self._encode_length(prompt) > self.config.max_model_len - 1024:
                        logger.warning(f"Prompt too long for {paper_name}, truncating")
                        # 截断内容
                        max_content_len = self.config.max_model_len - 3000
                        truncated_content = paper_content[:max_content_len]
                        generate_prompt = self.prompt_template.replace("{academic_paper}", truncated_content)
                        messages = [
                            {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                            {"role": "user", "content": generate_prompt}
                        ]
                        prompt = self._format_messages(messages)
                    
                    
                    inputs.append(prompt)
                    batch_metadata.append(paper_data)
                    
                except Exception as e:
                    logger.error(f"Error preparing input for {paper_data.get('paper_name', 'unknown')}: {e}")
                    stats["failed_generations"] += 1
                    continue
            
            if not inputs:
                continue
            
            # 批量生成
            try:
                outputs = self.llm.generate(inputs, self.sampling_params, use_tqdm=False)
                
                # 处理输出
                for idx, output in enumerate(outputs):
                    paper_data = batch_metadata[idx]
                    question_text = output.outputs[0].text
                    
                    # 解析问题列表
                    question_list = self._parse_questions(question_text)
                    
                    # 保存结果
                    result = paper_data.copy()
                    result["question_list"] = question_list
                    result["question_generation_time"] = datetime.now().isoformat()
                    result["model_used"] = self.model_name
                    
                    # 写入输出文件
                    with open(jsonl_file_output, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    
                    if question_list:
                        stats["successful_generations"] += 1
                        stats["total_questions"] += len(question_list)
                    else:
                        stats["failed_generations"] += 1
                        logger.warning(f"No questions generated for {paper_data['paper_name']}")
                        
            except Exception as e:
                logger.error(f"Error in batch generation: {e}")
                stats["failed_generations"] += len(batch_metadata)
        
        # 更新统计
        stats["processing_time"] = time.time() - start_time
        self.stats["generated_questions"] = stats["total_questions"]
        
        logger.info(f"Question generation completed: {stats}")
        return stats

    def _parse_questions(self, text: str) -> List[str]:
        """
        解析生成的问题文本
        
        Args:
            text: 生成的文本
            
        Returns:
            list: 问题列表
        """
        questions = []
        
        # 首先尝试从</think>标签后提取
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        
        # 查找问题模式
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # 匹配 [[1]], [[2]], [[3]] 格式
            match = re.match(r'\[\[(\d+)\]\]\s*(.+)', line)
            if match:
                question = match.group(2).strip()
                if question:
                    questions.append(question)
        
        # 如果没有找到标准格式，尝试其他格式
        if not questions:
            for i, line in enumerate(lines):
                line = line.strip()
                # 匹配 1. 2. 3. 格式
                match = re.match(r'^(\d+)\.\s*(.+)', line)
                if match:
                    question = match.group(2).strip()
                    if question and len(question) > 10:
                        questions.append(question)
        
        return questions

    def generate_answers(self, qa_data_file: str, output_file: str, use_cot: bool = True) -> Dict[str, Any]:
        """
        为问题生成答案
        
        Args:
            qa_data_file: 包含问题的JSON文件路径
            output_file: 输出文件路径
            use_cot: 是否使用Chain of Thought方式
            
        Returns:
            dict: 生成结果统计
        """
        start_time = time.time()
        stats = {
            "total_processed": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "no_answer_count": 0
        }
        
        # 读取QA数据
        qa_items = []
        with open(qa_data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                qa_items = data
            else:
                logger.error("Invalid QA data format")
                return stats
        
        logger.info(f"Found {len(qa_items)} QA items for answer generation")
        stats["total_processed"] = len(qa_items)
        
        # 准备答案生成提示模板
        if use_cot:
            answer_template = """你是一名半导体显示领域的专家，你需要回答的问题是：{question}

请注意：如果问题的难度超过你的能力，你无法确定回答准确性，请输出'无法作答!'。

Context: {context}

使用上述给定的上下文，回答问题。注意：
- 首先，请提供有关如何回答问题的详细 reasoning。
- 在 reasoning 中，如果需要复制上下文中的某些句子，请将其包含在 ##begin_quote## 和 ##end_quote## 中。这意味着 ##begin_quote## 和 ##end_quote## 之外的内容不是直接从上下文中复制的。
- 结束你的回答，以 final answer 的形式 <ANSWER>: $answer，答案应该简洁。

你必须以<Reasoning>: 开头，包含 reasoning 相关的内容；以 <ANSWER>: 开头，包含答案。"""
        else:
            answer_template = """{
    "instruction":"你是一个半导体显示领域的资深专家，你掌握TFT、OLED、LCD、QLED、EE、Design等显示半导体显示领域内的相关知识。请根据输入中的切片信息和问题进行回答。切片信息是可能相关的资料，切片信息的内容庞杂，不一定会包含目标答案，请仔细阅读每个切片后再作答，不得出现错误。",
    "input": {
        "context": "{context}",
        "question": "{question}"
    },
    "output": {
        "answer": "根据切片中提供的有效信息对问题进行详尽的回答，推荐分点回答格式。"
    },
    "requirements": {
        "criteria": "根据提供的切片信息提取有效信息进行回答",
        "format": "输出内容必须用中文作答。",
        "reasoning" : "在系统内部的think推理过程中，请将参考用到的上下文内容包含在 ##begin_quote## 和 ##end_quote## 中。"
    }
}"""
        
        # 批处理生成答案
        batches = self.to_batch(qa_items, self.batch_size)
        results = []
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Generating answers")):
            inputs = []
            batch_metadata = []
            
            for qa_item in batch:
                try:
                    question = qa_item.get("question", "")
                    context = qa_item.get("context", qa_item.get("paper_content", ""))
                    
                    if not question or not context:
                        logger.warning(f"Missing question or context for item: {qa_item.get('paper_name', 'unknown')}")
                        continue
                    
                    # 构建答案生成提示
                    prompt_content = answer_template.format(
                        question=question,
                        context=context
                    )
                    
                    messages = [
                        {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                        {"role": "user", "content": prompt_content}
                    ]
                    
                     # 应用聊天模板
                    prompt = self._format_messages(messages)
                    
                    # 检查长度
                    if self._encode_length(prompt) > self.config.max_model_len - 1024:
                        logger.warning(f"Prompt too long, truncating context")
                        # 截断上下文
                        max_context_len = self.config.max_model_len - 3000
                        truncated_context = context[:max_context_len]
                        prompt_content = answer_template.format(
                            question=question,
                            context=truncated_context
                        )
                        messages = [
                            {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                            {"role": "user", "content": prompt_content}
                        ]
                        
                        prompt = self._format_messages(messages)
                    
                    
                    inputs.append(prompt)
                    batch_metadata.append(qa_item)
                    
                except Exception as e:
                    logger.error(f"Error preparing input: {e}")
                    stats["failed_generations"] += 1
                    continue
            
            if not inputs:
                continue
            
            # 批量生成答案
            try:
                outputs = self.llm.generate(inputs, self.sampling_params, use_tqdm=False)
                
                # 处理输出
                for idx, output in enumerate(outputs):
                    qa_item = batch_metadata[idx]
                    answer_text = output.outputs[0].text
                    
                    # 解析答案
                    if use_cot:
                        # 提取COT格式的答案
                        answer = self._parse_cot_answer(answer_text)
                    else:
                        # 直接使用生成的答案
                        answer = answer_text.strip()
                    
                    # 检查是否无法作答
                    if "无法作答" in answer:
                        stats["no_answer_count"] += 1
                    
                    # 构建结果
                    result = qa_item.copy()
                    result["answer"] = answer
                    result["answer_generation_time"] = datetime.now().isoformat()
                    result["answer_model"] = self.model_name
                    result["use_cot"] = use_cot
                    
                    results.append(result)
                    stats["successful_generations"] += 1
                    
            except Exception as e:
                logger.error(f"Error in batch answer generation: {e}")
                stats["failed_generations"] += len(batch_metadata)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 更新统计
        stats["processing_time"] = time.time() - start_time
        logger.info(f"Answer generation completed: {stats}")
        
        return stats
    
    def _parse_cot_answer(self, text: str) -> str:
        """
        解析COT格式的答案
        
        Args:
            text: 生成的文本
            
        Returns:
            str: 提取的答案
        """
        # 查找<ANSWER>:标记
        answer_match = re.search(r'<ANSWER>:\s*(.+?)(?:\n|$)', text, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # 如果没有找到标准格式，尝试其他格式
        # 查找最后一段作为答案
        paragraphs = text.strip().split('\n\n')
        if paragraphs:
            last_paragraph = paragraphs[-1].strip()
            if last_paragraph and not last_paragraph.startswith('<Reasoning>'):
                return last_paragraph
        
        # 如果还是没有找到，返回整个文本
        return text.strip()

    async def judge_question_data(self, jsonl_file_input: str, save_paths: List[str]) -> Dict[str, Any]:
        """
        评估生成的问题质量
        
        Args:
            jsonl_file_input: 输入JSONL文件路径
            save_paths: 保存路径列表
            
        Returns:
            dict: 评估结果统计
        """
        start_time = time.time()
        stats = {
            "total_questions": 0,
            "passed_questions": 0,
            "failed_questions": 0,
            "error_questions": 0
        }
        
        # 读取输入数据
        questions_to_evaluate = []
        with open(jsonl_file_input, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("stats") == 1 and "question_li" in data:
                        questions_to_evaluate.append(data)
                except Exception as e:
                    logger.error(f"Error parsing line: {e}")
                    continue
        
        logger.info(f"Found {len(questions_to_evaluate)} questions to evaluate")
        stats["total_questions"] = len(questions_to_evaluate)
        
        # 批处理评估
        batches = self.to_batch(questions_to_evaluate, self.batch_size)
        final_results = []
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating questions")):
            evaluator_inputs = []
            batch_metadata = []
            
            # 准备输入
            for question_data in batch:
                try:
                    paper_content = question_data["paper_content"]
                    question_li = question_data["question_li"]
                    
                    # 构建评估提示
                    evaluator_prompt = self.evaluator_template.replace(
                        "{academic_paper}", paper_content
                    ).replace(
                        "{academic_question}", question_li
                    )
                    
                    evaluator_messages = [
                        {"role": "system", "content": "你是一个乐于助人的半导体显示技术领域的专家。"},
                        {"role": "user", "content": evaluator_prompt}
                    ]
                    
                    # 应用聊天模板
                    prompt = self._format_messages(evaluator_messages)
                     
                    evaluator_inputs.append(prompt)
                    batch_metadata.append(question_data)
                    
                except Exception as e:
                    logger.error(f"Error preparing evaluation input: {e}")
                    stats["error_questions"] += 1
                    continue
            
            if not evaluator_inputs:
                continue
            
            # 批量评估
            try:
                outputs = self.llm.generate(evaluator_inputs, self.sampling_params, use_tqdm=False)
                
                # 处理输出
                for idx, output in enumerate(outputs):
                    question_data = batch_metadata[idx]
                    evaluator_text = output.outputs[0].text.strip()
                    
                    # 提取评估结果
                    lines = evaluator_text.split('\n')
                    final_judgment = lines[-1] if lines else ""
                    
                    # 判断是否通过
                    is_passed = '【是】' in final_judgment
                    
                    if is_passed:
                        # 构建最终结果
                        result = {
                            "id": question_data["paper_name"],
                            "paper_content": question_data["paper_content"],
                            "question_li": question_data["question_li"],
                            "evaluation_result": evaluator_text,
                            "passed": True,
                            "timestamp": datetime.now().isoformat()
                        }
                        final_results.append(result)
                        stats["passed_questions"] += 1
                    else:
                        stats["failed_questions"] += 1
                        
            except Exception as e:
                logger.error(f"Error in batch evaluation: {e}")
                stats["error_questions"] += len(batch_metadata)
        
        # 保存结果
        for save_path in save_paths:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 如果文件已存在，追加结果
            existing_results = []
            if os.path.exists(save_path):
                try:
                    with open(save_path, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading existing results: {e}")
            
            # 合并结果
            all_results = existing_results + final_results
            
            # 保存
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Evaluation results saved to {save_path}")
        
        # 更新统计
        stats["processing_time"] = time.time() - start_time
        self.stats["passed_question_quality"] = stats["passed_questions"]
        
        logger.info(f"Question evaluation completed: {stats}")
        return stats

    def convert_questionlist_li_data(self, jsonl_file_questionlist: str, jsonl_file_li: str):
        """
        转换问题列表格式，将每个问题单独保存为一条记录
        
        Args:
            jsonl_file_questionlist: 输入文件路径（包含问题列表）
            jsonl_file_li: 输出文件路径（每个问题一条记录）
        """
        logger.info(f"Converting question list from {jsonl_file_questionlist} to {jsonl_file_li}")
        
        converted_count = 0
        error_count = 0
        
        with open(jsonl_file_questionlist, 'r', encoding='utf-8') as infile, \
             open(jsonl_file_li, 'w', encoding='utf-8') as outfile:
            
            for idx, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析原始数据
                    original_data = json.loads(line)
                    
                    # 提取需要的字段
                    stats = original_data.get("stats", 0)
                    paper_name = original_data.get("paper_name", "")
                    paper_content = original_data.get("paper_content", "")
                    score_text = original_data.get("score_text", "")
                    question_list = original_data.get("question_list", [])
                    
                    # 为每个问题创建单独的记录
                    for question_li in question_list:
                        processed_data = {
                            "stats": stats,
                            "paper_name": paper_name,
                            "paper_content": paper_content,
                            "score_text": score_text,
                            "question_li": question_li,
                            "source_line": idx,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # 写入新文件
                        json.dump(processed_data, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        converted_count += 1
                    
                    logger.debug(f"Processed entry {idx} successfully, generated {len(question_list)} records")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {idx} JSON decode error: {e}")
                    error_count += 1
                except Exception as e:
                    logger.error(f"Error processing entry {idx}: {e}")
                    error_count += 1
        
        logger.info(f"Conversion completed! Converted {converted_count} questions, {error_count} errors")

    async def run_full_pipeline(self, raw_folders: List[str], save_paths: List[str], 
                               output_dir: str = "/mnt/data/MLLM/lilinfeng/code/rl_data/process_data") -> Dict[str, Any]:
        """
        运行完整的三步评估流程
        
        Args:
            raw_folders: 原始文件夹列表
            save_paths: 保存路径列表
            output_dir: 输出目录
            
        Returns:
            dict: 完整流程的统计结果
        """
        logger.info("Starting full semiconductor QA generation pipeline")
        start_time = time.time()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义中间文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        judge_output_path = os.path.join(output_dir, f"judge_txt_output_{timestamp}.jsonl")
        generate_output_path = os.path.join(output_dir, f"generate_question_output_{timestamp}.jsonl")
        generate_li_output_path = os.path.join(output_dir, f"generate_question_output_{timestamp}_converted.jsonl")
        
        # 步骤1：评估文本质量
        logger.info("Step 1: Judging text quality...")
        text_quality_stats = await self.judge_md_data(raw_folders, save_paths, judge_output_path)
        
        # 步骤2：生成问题
        logger.info("Step 2: Generating questions...")
        generation_stats = await self.generate_question_data(judge_output_path, generate_output_path)
        
        # 步骤3：转换问题格式
        logger.info("Step 3: Converting question format...")
        self.convert_questionlist_li_data(generate_output_path, generate_li_output_path)
        
        # 步骤4：评估问题质量
        logger.info("Step 4: Evaluating question quality...")
        evaluation_stats = await self.judge_question_data(generate_li_output_path, save_paths)
        
        # 汇总统计
        total_time = time.time() - start_time
        final_stats = {
            "pipeline_summary": {
                "total_processing_time": total_time,
                "total_files_processed": text_quality_stats["total_files"],
                "files_passed_text_quality": text_quality_stats["passed_files"],
                "total_questions_generated": generation_stats["total_questions"],
                "questions_passed_quality": evaluation_stats["passed_questions"],
                "success_rate": {
                    "text_quality_pass_rate": (
                        text_quality_stats["passed_files"] / text_quality_stats["total_files"] * 100
                        if text_quality_stats["total_files"] > 0 else 0
                    ),
                    "question_generation_success_rate": (
                        generation_stats["successful_generations"] / generation_stats["total_processed"] * 100
                        if generation_stats["total_processed"] > 0 else 0
                    ),
                    "question_quality_pass_rate": (
                        evaluation_stats["passed_questions"] / evaluation_stats["total_questions"] * 100
                        if evaluation_stats["total_questions"] > 0 else 0
                    )
                }
            },
            "text_quality_stats": text_quality_stats,
            "generation_stats": generation_stats,
            "evaluation_stats": evaluation_stats,
            "output_files": {
                "judge_output": judge_output_path,
                "generation_output": generate_output_path,
                "converted_output": generate_li_output_path,
                "final_results": save_paths
            }
        }
        
        # 保存汇总统计
        stats_path = os.path.join(output_dir, f"pipeline_stats_{timestamp}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Pipeline completed in {total_time:.2f} seconds")
        logger.info(f"Summary statistics saved to {stats_path}")
        
        return final_stats


# 便捷函数
async def run_semiconductor_qa_generation(
    raw_folders: list,
    save_paths: list,
    model_name: str = "qwq_32",
    batch_size: int = 4,
    gpu_devices: str = "0,1"
):
    """运行半导体QA生成 - 修复版本"""
    logger.info("初始化半导体QA生成器...")
    
    try:
        # 修复：正确创建生成器，不传递model_name参数
        generator = SemiconductorQAGenerator(
            batch_size=batch_size,
            gpu_devices=gpu_devices
        )
        
        # 通过属性设置模型名称
        generator.model_name = model_name
        logger.info(f"设置模型名称为: {model_name}")
        
        # 如果需要加载模型
        if hasattr(generator, 'load_model'):
            generator.load_model()
        
        logger.info("半导体QA生成器初始化完成")
        
        # 继续原有的处理逻辑...
        # 这里保持你原有的处理代码不变
        
    except Exception as e:
        logger.error(f"QA生成失败: {str(e)}")
        import traceback
        logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
        raise