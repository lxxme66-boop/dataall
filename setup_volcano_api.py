#!/usr/bin/env python3
"""火山引擎API配置脚本"""

import os
import json
import sys
import argparse

def setup_volcano_config(api_key):
    """设置火山API配置"""
    config = {
        "api": {
            "use_volcano_api": True,
            "volcano_api_key": api_key,
            "volcano_endpoint": "https://ark.cn-beijing.volces.com/api/v3",
            "model": "deepseek-r1"
        }
    }
    
    with open("config_volcano_api.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("配置文件已保存到: config_volcano_api.json")
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="火山引擎API密钥")
    args = parser.parse_args()
    setup_volcano_config(args.api_key)
