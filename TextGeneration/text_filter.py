# -*- coding: utf-8 -*-
"""
文本过滤模块 - 用于过滤学术论文中的无关内容
从半导体显示技术领域QA生成系统移植
"""
import re
import os


def is_to_drop(text):
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
    patterns = [r'http://www.cnki.net', r'https://www.cnki.net', r'^\[\d{1,4}\]', r'^\*\s+\[\d{1,4}\]', r'^\*\s+\(\d{1,4}\)', 
                r'^致谢.*[0-9]$', r'.*致\s*谢.*', r'.*目\s*录.*', r'\.\.\.\.\.\.\.\.', r'\…\…\…', r"(http://www|doi:|DOI:|please contact)",
                r"(work was supported by|study was supported by|China|Republic of Korea|Authorized licensed use limited to)",
                r"\s[1-9]\d{5}(?!\d)",  # 邮编
                r"\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*", r"(received in revised form|All rights reserved|©)", r"[a-zA-z]+://[^\s]*",
                r"(13[0-9]|14[5|7]|15[0|1|2|3|5|6|7|8|9]|18[0|1|2|3|5|6|7|8|9])\d{8}", r"\d{3}-\d{8}|\d{4}-\d{7}",
                r'^分\s*类\s*号', r'^学\s*科\s*专\s*业', r'^签\s*字\s*日\s*期', r'^申\s*请\s*人\s*员\s*姓\s*名',
                r'^日\s*期', r'^指\s*定\s*教\s*师', r'学\s*位\s*论\s*文', r'^工\s*作\s*单\s*位', r'^电\s*话', r'^通讯地址', r'^邮\s*编', 
                r'^中\s*图\s*分\s*类\s*号', r'^评\s*阅\s*人', r'^签\s*名', r'^分\s*类\s*号', r'^密\s*级', r'^学\s*号', r'^院\s*系', 
                r'^委\s*员', r'^国内图书分类号', r'^国际图书分类号', r'^导\s*师', r'^申\s*请\s*学\s*位', r'^工\s*程\s*领\s*域', r'^所\s*在\s*单\s*位', 
                r'^答\s*辩', r'^作\s*者', r'^专\s*业', r'^保\s*密', r'^不\s*保\s*密', r'^硕\s*土\s*姓\s*名', r'^导\s*师', r'^职\s*称', r'^声\s*明', 
                r'^申请学位', r'^学科、专业', r'^学\s*校\s*代\s*码', r'^邢\s*坤\s*太\s*学', r'^学\s*科\s*门\s*类', r'^培\s*养\s*院\s*系',
                r'^研\s*究\s*生', r'^专\s*业', r'^完\s*成\s*日\s*期', r'^年\s*月\s*日', r'^审\s*级', r'^单\s*位\s*代\s*码', r'^密\s*码', 
                r'^学\s*位\s*授\s*予', r'^校\s*址', r'^授\s*予', r'^论\s*文\s*分\s*类\s*号', r'^研\s*突\s*生', r'^研\s*究\s*方\s*向:', 
                r'^研\s*究\s*生', r'^学\s*校\s*代\s*号', r'^主\s*席', r'^U\s*D\s*C', r'^U.D.C', r'^论\s*文\s*起\s*止', r'^论\s*文\s*样\s*纸', 
                r'^完\s*成\s*时\s*间', r'^学\s*校\s*编\s*码', r'^声\s*明\s*人', r'^分\s*类\s*号', r'^培\s*养\s*单\s*位', r'^提\s*交\s*论\s*文', 
                r'^资\s*助', r'^学科(专业)', r'^提\s*交\s*日\s*期', r'^学\s*科\s*名\s*称', r'^课\s*题\s*人', r'^学\s*科\s*门\s*类', 
                r'^一\s*级\s*学\s*科', r'^学\s*位\s*申\s*请', r'^学\s*院\s*名\s*称', r'^主\s*任', r'^院\s*系', r'^专\s*业', r'^姓\s*名', 
                r'^完\s*成\s*日\s*期', r'^作\s*者', r'^申\s*请\s*学\s*位', r'^工\s*程\s*领\s*域', r'^学\s*科\s*名\s*称', r'^领\s*域', r'^学\s*院', 
                r'^提\s*交\s*日\s*期', r'^授\s*予\s*学\s*位', r'^学\s*科', r'^所\s*在\s*单\s*位', r'^电\s*子\s*邮\s*箱', r'^联\s*系\s*地\s*址',
                
                r'^!\[\]\(images/.*',  # 多余（可在检查有无中文字符时去掉）且导致报错
                
                r'^\[?\d+\]?', r'^\s*\[?\d+\]?', r'^\［?\d+\］?', r'^\s*\［?\d+\］?' # mineru解析的参考文献格式
                ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
        
    patterns = ['申请号|专利号|已录用|学报|研究生|已收录|攻读|第一作者|第二作者|参考文献|专业名称|863项目|导师',
                '教授|感谢|致谢|谢谢|指导|朋友|家人|亲友|师弟|师妹|老师|同学|父母|充实|答辩|祝愿|独创性声明|作者签名',
                '发表文章|论文使用授权声明|本人|知网|论文使用权|发表的论文|申请的专利|申请专利|发表的文章|发表学术论文|发表论文',
                '参与科研项目|作者简介|三年的学习|大学硕士学位论文|大学博士学位论文|涉密论文|学校代码|论文提交日期|委员：|中图分类号',
                '原创性声明|顺利完成学业|All rights reserved|参 考 文 献|参考文献|所在学院|国家自然科学基金|教育部重点学科建设',
                '时间飞梭|时光飞梭|光阴似箭|白驹过隙|论文版权|本学位论文|使用授权书|References|Acknowledgements',
                '论文著作权|保密的学位论文|中国第一所现代大学|参加科研情况|独 创 性 声 明|论文使用授权|获得的专利|家庭的爱|文献标识码|文章编号'
                ]
    for pattern in patterns:
        if re.findall(pattern, text):
            return True   
        
    """
    判断是否不包含中文字符（暂时把公式也去掉）
    """
    num = 0
    for t in text:
        if  '\u4e00' <= t <= '\u9fa5':
            num += 1    
    if num / len(text) < 0.01:
        return True
                
    return False


def drop(texts, concatenation="\n"):
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
        if not is_to_drop(text):
            new_texts.append(text)
    return concatenation.join(new_texts)


def load_paper(file_path):
    """
    加载论文并进行过滤处理
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 过滤后的论文内容
    """
    with open(file_path, "r", encoding="utf8") as f:
        content = f.read()
    deal_content = drop(content)
    return deal_content


def to_batch(lst, groupsize):
    """
    将列表分批处理
    [a,b,c,d,e] -> [[a,b], [c,d], [e]], for batch inference
    
    Args:
        lst: 输入列表
        groupsize: 每批大小
        
    Returns:
        list: 分批后的列表
    """
    return [lst[i:i+groupsize] for i in range(0, len(lst), groupsize)]