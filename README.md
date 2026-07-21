# Skill-chat

## 环境配置：
(不完整)

```bash
pip install graphrag==2.7.2
```

## GraphRAG package tutotial:
https://microsoft.github.io/graphrag/

## 相关命令：
 1. graphrag init --root PATH
 2. graphrag index --root PATH
 
## 操作说明：

初始化之后，root目录下会有prompts目录和settings.yaml文件，里面的内容都是官方默认的内容，
用 [ESConv](ESConv)里面的文件进行替换，修改的内容主要包括：
1. prompts目录下的文件包括抽取entities的prompts,进行社区报告总结的prompts等
2. settings.yaml文件，比较重要，里面有llm服务的地址，embedding服务的地址，input的文件地址等，

配置好之后直接执行graphrag index --root PATH 即可。
