"""
Model Rewrite Module

This module provides data rewriting and enhancement capabilities for QA generation.
It includes functionality for:
- Data augmentation and paraphrasing
- Quality improvement of existing QA pairs
- Multi-modal data enhancement
- Professional domain-specific rewriting
"""

from .data_generation import DataRewriter, generate_rewritten_data
from .data_label import DataLabeler, quality_check_data
from .prompt_builder import PromptBuilder, build_rewrite_prompt

__all__ = [
    'DataRewriter',
    'generate_rewritten_data', 
    'DataLabeler',
    'quality_check_data',
    'PromptBuilder',
    'build_rewrite_prompt'
]

__version__ = "1.0.0"