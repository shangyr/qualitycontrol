
API_KEY = ''
HOST = 'https://api.f2gpt.com/v1/chat/completions'

import requests
import copy
# 速度慢 

class ChatBot:
    def __init__(self, api_key = None, host = None) -> None:
        
        if api_key is not None: self.api_key = api_key
        else: self.api_key = API_KEY
        if host is not None: self.host = host
        else: self.host = HOST
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": ""}],
            "temperature": 0.7,
        }

    def chat(self, sentence):
        """
        chat with chatgpt
        """
        sentence = sentence + self.prompt

        data = copy.deepcopy(self.data)
        data['messages'][0]['content'] = sentence
        response = requests.post(self.host, headers=self.headers, json=data)

        # print(response.status_code)
        # print(response.json())

        return self.parser(response.json()['choices'][0]['message']['content'])
    
    def parser(self, res):
        triples = res.strip().split('\n')
        triples = [t.split('-') for t in triples]
        return triples

    @property
    def prompt(self):
        # return "请提取出这段话所有关系，以三元组的形式返回，格式为：实体1-关系-实体2"
        return "请提取出这段话所有医疗实体，涉及的实体类型均在这个集合中\{疾病、症状、检查\}，格式为 实体名称-实体类型"

if __name__ == '__main__':
    c = ChatBot()
    triples = c.chat('(3)体温下降期：由于病因的消除，致热原的作用逐渐减弱或消失，体温中枢的体温调定点逐渐降至正常水平，产热相对减少，散热大于产热，使体温降至正常水平')
    print(triples)
