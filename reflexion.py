from openai import OpenAI
from datasets import load_dataset
import re


client = OpenAI(api_key="sk-10a002db22ac4be4846cd9f800939590", base_url="https://api.deepseek.com")

def get_response(messages):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    return response.choices[0].message.content

def answer_the_question(question, reflections):
    
    instructions = "You are a helpful assistant."
    
    user_content = f"Reflections: {reflections}\n```Question: {question}``` Let's think step by step."
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content}
    ]
    response = get_response(messages)

    instructions = "You are a helpful assistant."
    user_content = f"```Question: {question}``` Let's think step by step.\nAnswer: {response}"
    results = user_content
    user_content += "\nSo your calculated number is:"
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content}
    ]
    response = get_response(messages)

    return response, results

def get_reflection(unsuccessful_example):
    instructions = "You are a helpful assistant. Please reflect on the following example."
    user_content = f"{unsuccessful_example}"
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_content}
    ]
    response = get_response(messages)
    return response

class Reflexion:
    def __init__(self, question, true_answer):
        self.question = question
        self.true_answer = true_answer
        self.unsuccessful_examples = []
        self.reflection_str = ''
        self.succeed = False
        self.n_trials = 0
        self.uncertain_ans = None
    
    def run(self):
        if self.succeed:
            return
        if self.n_trials > 0 and not self.succeed:
            self.reflect()
        
        answer, results = answer_the_question(self.question, self.reflection_str)
        if self.true_answer in answer:
            self.succeed = True
        else:
            self.succeed = False
        
        self.uncertain_ans = answer
        if not self.succeed:
            self.unsuccessful_examples.append(results + "\nYour answer is incorrect.")
        self.n_trials += 1

    def reflect(self):
        self.reflection_str += "\n" + get_reflection(self.unsuccessful_examples[-1])

    def is_correct(self, answer):
        if self.true_answer in answer:
            return True
        return False
    
    def to_dict(self):
        return {
            'question': self.question,
            'ground_truth_answer': self.true_answer,
            'predicted_answer': self.uncertain_ans,
            'succeed': self.succeed,
            'unsuccessful_examples': self.unsuccessful_examples,
            'reflection': self.reflection_str
        }


if __name__ == '__main__':
    
    dataset = load_dataset('parquet', data_files='gsm8k/test-00000-of-00001.parquet')
    dataset = dataset['train']
   

    total_samples = 0
    correct_samples = 0
    max_trials = 3
    

    for i, sample in enumerate(dataset):
        
        a = sample['answer']
        a1 = re.findall(r'#### (.+)', a)
        a1 = ''.join(a1)

        agent = Reflexion(sample['question'], a1)

        for _ in range(max_trials):
            agent.run()
        total_samples += 1
        if agent.succeed:
            correct_samples += 1

        # 把编号和是否正确写到CSV里
        with open('reflexion_result.csv', 'a') as f:
            f.write(f"{i},{int(agent.succeed)}\n")

        accuracy = correct_samples / total_samples
        print(f"Progress: {i+1} | Accuracy: {accuracy:.4f}")


    print(f"Reflexion Accuracy: {correct_samples/total_samples}")