#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简历DOCX文件预提取模块 - 人才评价模型项目

功能：
1. 解析.docx格式的简历文件，提取文本内容
2. 保留基本格式结构信息（标题、段落、列表等）
3. 初步的文本规范化处理，为后续清洗做准备
4. 错误处理与日志记录
5. 与数据清洗模块的无缝集成

修复记录（v2.0）：
  P0 - integrate_with_cleaner 候选人ID解析容错：当文件名不含数字（如纯汉字文件名）时，
       原逻辑回退至 JD 描述文本，产生 "or生产总监" 等脏数据。
       修复方案：无数字时使用文件名 MD5 哈希前8位生成稳定唯一 ID，
       并在元数据中记录 id_source = 'hash_fallback' 供下游审计追踪。
"""

import os
import re
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table, _Row, _Cell
from docx.text.paragraph import Paragraph


def setup_logging():
    """
    配置统一的日志系统

    功能：
    1. 创建项目统一的日志文件夹
    2. 配置日志文件的命名规范
    3. 实现日志文件的轮转策略
    4. 确保异常情况下的日志记录
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, 'logs')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"创建日志目录: {log_dir}")

    log_file_prefix = 'docx_extractor'
    current_date = datetime.now().strftime('%Y%m%d')
    log_file_name = f"{log_file_prefix}_{current_date}.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    logger_obj = logging.getLogger(__name__)
    logger_obj.setLevel(logging.INFO)

    if logger_obj.handlers:
        logger_obj.handlers.clear()

    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    logger_obj.addHandler(file_handler)
    logger_obj.addHandler(console_handler)

    logger_obj.info(f"日志系统初始化完成，日志文件: {log_file_path}")

    return logger_obj, log_dir


logger, LOG_DIR = setup_logging()


def get_logger():
    """获取日志记录器"""
    return logging.getLogger(__name__)


class DocxExtractor:
    """
    DOCX文件解析器，用于从.docx文件中提取文本内容
    """
    
    def __init__(self):
        """初始化DOCX解析器"""
        pass
    
    def extract_from_file(self, file_path: str) -> Dict:
        """
        从.docx文件中提取文本内容
        
        Args:
            file_path: .docx文件的路径
            
        Returns:
            Dict: 包含提取结果的字典，包括：
                - text: 提取的文本内容
                - structured_data: 结构化数据（包含格式信息）
                - metadata: 文件元数据
                - status: 提取状态（success/failure）
                - error: 错误信息（如果有）
        """
        result = {
            'text': '',
            'structured_data': [],
            'metadata': {},
            'status': 'success',
            'error': None
        }
        
        try:
            # 检查文件路径
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            if not file_path.endswith('.docx'):
                raise ValueError(f"文件格式错误，仅支持.docx文件: {file_path}")
            
            # 读取文件
            doc = Document(file_path)
            
            # 提取元数据
            result['metadata'] = self._extract_metadata(doc, file_path)
            
            # 提取文本和结构
            text, structured_data = self._extract_content(doc)
            result['text'] = text
            result['structured_data'] = structured_data
            
            # 初步规范化处理
            result['text'] = self._normalize_text(result['text'])
            
            logger.info(f"成功提取文件: {file_path}")
            
        except FileNotFoundError as e:
            result['status'] = 'failure'
            result['error'] = str(e)
            logger.error(f"文件不存在: {file_path}")
        except ValueError as e:
            result['status'] = 'failure'
            result['error'] = str(e)
            logger.error(f"文件格式错误: {file_path}")
        except Exception as e:
            result['status'] = 'failure'
            result['error'] = f"提取过程出错: {str(e)}"
            logger.error(f"提取文件时出错: {file_path}, 错误: {str(e)}")
        
        return result
    
    def _extract_metadata(self, doc: DocxDocument, file_path: str) -> Dict:
        """
        提取文件元数据
        
        Args:
            doc: DOCX文档对象
            file_path: 文件路径
            
        Returns:
            Dict: 元数据字典
        """
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'num_paragraphs': len(doc.paragraphs),
            'num_tables': len(doc.tables)
        }
        
        # 尝试提取文档属性
        try:
            core_props = doc.core_properties
            if core_props.title:
                metadata['title'] = core_props.title
            if core_props.author:
                metadata['author'] = core_props.author
            if core_props.created:
                metadata['created'] = core_props.created.isoformat()
        except Exception as e:
            logger.warning(f"提取文档属性时出错: {str(e)}")
        
        return metadata
    
    def _extract_content(self, doc: DocxDocument) -> Tuple[str, List]:
        """
        提取文档内容，包括文本和结构信息
        
        Args:
            doc: DOCX文档对象
            
        Returns:
            Tuple[str, List]: (提取的文本, 结构化数据)
        """
        text_parts = []
        structured_data = []
        
        for i, element in enumerate(doc.element.body):
            if isinstance(element, CT_P):
                # 处理段落
                para = Paragraph(element, doc)
                para_text = para.text.strip()
                if para_text:
                    text_parts.append(para_text)
                    structured_data.append({
                        'type': 'paragraph',
                        'text': para_text,
                        'style': para.style.name if para.style else None,
                        'index': i
                    })
            elif isinstance(element, CT_Tbl):
                # 处理表格
                table = Table(element, doc)
                table_text = self._extract_table_content(table)
                if table_text:
                    text_parts.append(table_text)
                    structured_data.append({
                        'type': 'table',
                        'text': table_text,
                        'num_rows': len(table.rows),
                        'num_cols': len(table.columns),
                        'index': i
                    })
        
        full_text = '\n'.join(text_parts)
        return full_text, structured_data
    
    def _extract_table_content(self, table: Table) -> str:
        """
        提取表格内容
        
        Args:
            table: 表格对象
            
        Returns:
            str: 表格内容的文本表示
        """
        table_text = []
        
        for row_idx, row in enumerate(table.rows):
            row_text = []
            for cell_idx, cell in enumerate(row.cells):
                cell_text = cell.text.strip()
                row_text.append(cell_text)
            if any(row_text):
                table_text.append('\t'.join(row_text))
        
        return '\n'.join(table_text)
    
    def _normalize_text(self, text: str) -> str:
        """
        初步规范化文本
        
        Args:
            text: 原始文本
            
        Returns:
            str: 规范化后的文本
        """
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 去除多余的空白字符
        text = ' '.join(text.split())
        
        # 去除首尾空白
        text = text.strip()
        
        # 统一全角字符为半角
        full_to_half = {
            '，': ',', '。': '.', '！': '!', '？': '?',
            '；': ';', '：': ':', '“': '"', '”': '"',
            '‘': "'", '’': "'", '（': '(', '）': ')',
            '【': '[', '】': ']', '—': '-', '…': '...'
        }
        for full, half in full_to_half.items():
            text = text.replace(full, half)
        
        return text
    
    def extract_from_files(self, file_paths: List[str]) -> List[Dict]:
        """
        批量提取多个.docx文件
        
        Args:
            file_paths: .docx文件路径列表
            
        Returns:
            List[Dict]: 提取结果列表
        """
        results = []
        
        for file_path in file_paths:
            result = self.extract_from_file(file_path)
            results.append(result)
        
        return results


def integrate_with_cleaner(extractor: DocxExtractor, cleaner, file_path: str, industry: str = ""):
    """
    将DOCX提取器与数据清洗模块集成
    
    Args:
        extractor: DocxExtractor实例
        cleaner: ResumeDataCleaner实例
        file_path: .docx文件路径
        industry: 行业信息
        
    Returns:
        ResumeData: 清洗后的简历数据
    """
    # 提取DOCX内容
    extraction_result = extractor.extract_from_file(file_path)
    
    if extraction_result['status'] == 'failure':
        logger.error(f"提取失败: {extraction_result['error']}")
        # 返回空的ResumeData对象
        return cleaner.clean_single('', industry=industry)
    
    # 使用提取的文本进行清洗
    text = extraction_result['text']
    metadata = extraction_result['metadata']
    
    # 从文件名或元数据中提取候选人ID（如果可能）
    candidate_id = ''
    id_source = 'unknown'
    file_name = metadata.get('file_name', '')

    # 尝试从文件名中提取数字ID
    id_match = re.search(r'\d+', file_name)
    if id_match:
        candidate_id = id_match.group(0)
        id_source = 'filename_digit'
    else:
        # 哈希回退：纯汉字或无数字文件名时，用文件名 MD5 生成稳定 ID
        # 取前8位大写十六进制，确保唯一性且可追溯
        candidate_id = hashlib.md5(file_name.encode('utf-8')).hexdigest()[:8].upper()
        id_source = 'hash_fallback'
        logger.warning(
            f"文件名无数字ID，使用哈希回退生成稳定ID: '{file_name}' → '{candidate_id}'"
        )

    # 将 id_source 写入元数据，供下游系统审计追踪
    metadata['id_source'] = id_source
    
    # 调用清洗器
    cleaned_data = cleaner.clean_single(text, industry=industry, candidate_id=candidate_id)
    
    # 将元数据添加到清洗结果中（如果需要）
    # 注意：这里需要根据ResumeData的结构进行调整
    # 例如，如果ResumeData有metadata字段，可以添加
    # cleaned_data.metadata = metadata
    
    return cleaned_data


# 示例用法
if __name__ == '__main__':
    # 导入数据清洗模块
    from resume_data_cleaner import ResumeDataCleaner
    
    # 创建提取器和清洗器
    extractor = DocxExtractor()
    cleaner = ResumeDataCleaner()
    
    # 示例文件路径（请替换为实际的.docx文件路径）
    sample_docx = "D:/project_TalentSystem/data/branding/20230524-HFF（）for-品牌总经理.docx"
    
    if os.path.exists(sample_docx):
        # 提取并清洗
        result = integrate_with_cleaner(extractor, cleaner, sample_docx, industry='信息技术')
        
        # 输出结果
        print("=" * 60)
        print("DOCX提取与清洗结果")
        print("=" * 60)
        print(f"目标岗位: {result.target_position}")
        print(f"行业: {result.industry}")
        print(f"学历等级: {result.education_level.value}")
        print(f"工作经历数: {len(result.work_experiences)}")
        print(f"软素质得分 - 领导力: {result.leadership_score}")
        print(f"软素质得分 - 情商: {result.eq_score}")
    else:
        print(f"示例文件不存在: {sample_docx}")
        print("请创建一个示例简历.docx文件进行测试")
