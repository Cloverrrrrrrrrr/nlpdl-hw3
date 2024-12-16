# import os
# import torch
# from datasets import load_dataset
# from openai import OpenAI
# from nltk.translate.bleu_score import sentence_bleu   
# import re
# import time

# def naive_prompt(question):
#     prompt = f"Answer the following question: {question}"
#     return prompt

# def cot_prompt(question):
#     prompt = f"Let's break this down step by step to solve the following question: {question}"
#     return prompt

# def icl_prompt(context, question):
#     prompt = f"Given the following example:\n{context}\nNow, answer this question: {question}"
#     return prompt



# def generate_answer_with_deepseek(query):
#     # 合并检索到的文档和用户输入作为生成的上下文
#     context = query
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": context}
#         ],
#         stream=False
#     )
#     model_ans = response.choices[0].message.content

#     final_message = "So your calculated number is:"

#     response2 = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": context},
#             {"role": "assistant", "content": model_ans},
#             {"role": "user", "content": final_message}
#         ],
#         stream=False
#     )

#     return response2.choices[0].message.content



# dataset = load_dataset('parquet', data_files='gsm8k/test-00000-of-00001.parquet')

# # 应用上面的函数



# client = OpenAI(api_key="sk-10a002db22ac4be4846cd9f800939590", base_url="https://api.deepseek.com")

# dataset2 = load_dataset('parquet', data_files='gsm8k/train-00000-of-00001.parquet')

# # 遍历整个数据集中的问题，生成答案，计算整体的准确率
# total = 0
# correct1 = 0
# correct2 = 0
# correct3 = 0
# print(len(dataset['train']))

# for i in range(10):
#     t = time.time()
#     prompt1 = naive_prompt(dataset['train'][i]['question'])
#     prompt2 = cot_prompt(dataset['train'][i]['question'])
#     prompt3 = icl_prompt(dataset2['train'][0], dataset['train'][i]['question'])
#     r1 = generate_answer_with_deepseek(prompt1)
#     r2 = generate_answer_with_deepseek(prompt2)
#     r3 = generate_answer_with_deepseek(prompt3)
#     a = dataset['train'][i]['answer']
#     a1 = re.findall(r'#### (.+)', a)
#     a1 = ''.join(a1)
#     if a1 in r1:
#         correct1 += 1
#     if a1 in r2:
#         correct2 += 1
#     if a1 in r3:
#         correct3 += 1
#     total += 1
#     print(f"Time: {time.time() - t}")

# print(f"Accuracy for Naive Prompt: {correct1 / total}")
# print(f"Accuracy for COT Prompt: {correct2 / total}")
# print(f"Accuracy for ICL Prompt: {correct3 / total}")

import os
import torch
from datasets import load_dataset
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu   
import re
import time
import concurrent.futures  # 导入并发库

def naive_prompt(question):
    prompt = f"Answer the following question: {question}"
    return prompt

def cot_prompt(question):
    prompt = f"Answer the following question: {question}\nLet's think step by step."
    return prompt

def icl_prompt(context, question):
    prompt = f"Given the following examples:\n{context}\nNow, answer this question: {question}"
    return prompt

def fs_prompt(context, question):
    prompt = f"Given the following examples:\n{context}\nNow, answer this question: {question}\nLet's think step by step."
    return prompt


def generate_answer_with_deepseek(query):
    """调用 DeepSeek API 生成答案的函数"""
    context = query
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context}
        ],
        stream=False
    )
    model_ans = response.choices[0].message.content

    final_message = "So your calculated number is:"

    response2 = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context},
            {"role": "assistant", "content": model_ans},
            {"role": "user", "content": final_message}
        ],
        stream=False
    )


    return response2.choices[0].message.content


def process_query(i, dataset, dataset2):
    """并发处理每个问题的查询并返回结果"""
    prompt1 = naive_prompt(dataset['train'][i]['question'])
    prompt2 = cot_prompt(dataset['train'][i]['question'])
    prompt3 = icl_prompt(dataset2, dataset['train'][i]['question'])
    prompt4 = fs_prompt(dataset2, dataset['train'][i]['question'])

    r1 = generate_answer_with_deepseek(prompt1)
    r2 = generate_answer_with_deepseek(prompt2)
    r3 = generate_answer_with_deepseek(prompt3)
    r4 = generate_answer_with_deepseek(prompt4)
    
    a = dataset['train'][i]['answer']
    a1 = re.findall(r'#### (.+)', a)
    a1 = ''.join(a1)

    return r1, r2, r3, r4, a1


def calculate_accuracy(results):
    """计算准确率"""
    total = len(results)
    
    #把正确与否（1或0）按照i储存在csv里
    with open('icl.csv', 'w') as f:
        for i in range(total):# 左边i右边1或0
            f.write(f"{i},{1 if results[i][1] in results[i][0] else 0}\n")
            
    correct1 = sum([1 for r1, _, _, _, a1 in results if a1 in r1])
    correct2 = sum([1 for _, r2, _, _, a1 in results if a1 in r2])
    correct3 = sum([1 for _, _, r3, _, a1 in results if a1 in r3])
    correct4 = sum([1 for _, _, _, r4, a1 in results if a1 in r4])

    print(f"Accuracy for Naive Prompt: {correct1 / total}")
    print(f"Accuracy for COT Prompt: {correct2 / total}")
    print(f"Accuracy for ICL Prompt: {correct3 / total}")
    print(f"Accuracy for few-shot CoT Prompt: {correct4 / total}")


# 加载数据集
dataset = load_dataset('parquet', data_files='gsm8k/test-00000-of-00001.parquet')

#使用dataset2的前三个问题作为ICL的例子

icl_example = [
    "Example 1: \nInput: 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?'\nOutput: 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72'",
    "Example 2: \nInput: 'Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?'\nOutput: 'Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10'",
    "Example 3: \nInput: 'Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?'\nOutput: 'In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n#### 5'"
]




# 初始化 DeepSeek API 客户端
client = OpenAI(api_key="sk-10a002db22ac4be4846cd9f800939590", base_url="https://api.deepseek.com")

lens = len(dataset['train'])
# 并发处理问题
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(lambda i: process_query(i, dataset, icl_example), range(1)))
  

# 计算准确率
calculate_accuracy(results)


