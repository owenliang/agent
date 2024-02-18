import os
import json
from langchain_community.tools.tavily_search import TavilySearchResults
import requests

os.environ['TAVILY_API_KEY']='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'    # travily搜索引擎api key

def llm(query,history=[],user_stop_words=[]):    # 调用api_server
    response=requests.post('http://localhost:8000/chat',json={
        'query':query,
        'stream': False,
        'history':history,
        'user_stop_words':user_stop_words,
    })
    return response.json()['text']
    
# travily搜索引擎
tavily=TavilySearchResults(max_results=5)
tavily.description='在线搜索知识的引擎，可以检索到最近或者当下的实时数据，当用户的提问可能涉及实时信息的时候，你应该使用该工具。'

# 工具列表
tools=[tavily, ]  

tool_names='or'.join([tool.name for tool in tools])  # 拼接工具名
tool_descs=[] # 拼接工具详情
for t in tools:
    args_desc=[]
    for name,info in t.args.items():
        args_desc.append({'name':name,'description':info['description'] if 'description' in info else '','type':info['type']})
    args_desc=json.dumps(args_desc,ensure_ascii=False)
    tool_descs.append('%s: %s,args: %s'%(t.name,t.description,args_desc))
tool_descs='\n'.join(tool_descs)

#print(tool_names)
#print(tool_descs)

prompt_tpl='''Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

You may need the following infomation as well: 
{chat_history}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
{agent_scratchpad}
'''

# 测试prompt
prompt=prompt_tpl.format(chat_history='',tool_descs=tool_descs,tool_names=tool_names,query='查一下明天青岛的天气',agent_scratchpad='')
#print(prompt)

def agent_execute(query,chat_history=[]):
    global tools,tool_names,tool_descs,prompt_tpl,llm,tokenizer
    
    agent_scratchpad='' # agent执行过程
    while True:
        # 1）触发llm思考下一步action
        history='\n'.join(['Question:%s\nAnswer:%s'%(his[0],his[1]) for his in chat_history])
        prompt=prompt_tpl.format(chat_history=history,tool_descs=tool_descs,tool_names=tool_names,query=query,agent_scratchpad=agent_scratchpad)
        print('prompt: %s\n\033[32magent正在思考... ...'%prompt,flush=True)
        response=llm(prompt,user_stop_words=['Observation:'])
        print('---LLM raw response\n%s\n\033[0m---'%response,flush=True)
        
        # 2）解析thought+action+action input+observation or thought+final answer
        thought_i=response.rfind('Thought:')
        final_answer_i=response.rfind('\nFinal Answer:')
        action_i=response.rfind('\nAction:')
        action_input_i=response.rfind('\nAction Input:')
        observation_i=response.rfind('\nObservation:')
        
        # 3）返回final answer，执行完成
        if final_answer_i!=-1 and thought_i<final_answer_i:
            print('\033[34mfinal answer: \033[0m%s\n\033[34m\n'%(response[final_answer_i+len('\nFinal Answer:'):]),flush=True)
            chat_history.append((query,response[final_answer_i+len('\nFinal Answer:'):]))
            return True,response[final_answer_i+len('\nFinal Answer:'):],chat_history
        
        # 4）解析action
        if not (thought_i<action_i<action_input_i<observation_i):
            print('\033[31m LLM回复格式异常, 放弃继续思考.\nresponse:%s\n \033[0m'%response,flush=True)
            return False,'LLM回复格式异常',chat_history
        thought=response[thought_i+len('Thought:'):action_i].strip()
        action=response[action_i+len('\nAction:'):action_input_i].strip()
        action_input=response[action_input_i+len('\nAction Input:'):observation_i].strip()
        print('\033[34mthought: \033[0m%s\n\033[34maction: \033[0m%s\n\033[34maction_input: \033[0m%s\n'%(thought,action,action_input),flush=True)
        
        # 5）匹配tool
        the_tool=None
        for t in tools:
            if t.name==action:
                the_tool=t
                break
        if the_tool is None:
            observation='the tool not exist'
            agent_scratchpad=agent_scratchpad+response+observation+'\n'
            continue 
        
        # 6）执行tool
        try:
            action_input=json.loads(action_input)
            tool_ret=the_tool.invoke(input=json.dumps(action_input))
        except Exception as e:
            observation='the tool has error:{}'.format(e)
        else:
            observation=str(tool_ret)
        agent_scratchpad=agent_scratchpad+response+observation+'\n'

def agent_execute_with_retry(query,chat_history=[],retry_times=3):
    for i in range(retry_times):
        success,result,chat_history=agent_execute(query,chat_history=chat_history)
        if success:
            return success,result,chat_history
    return success,result,chat_history

my_history=[]
while True:
    query=input('query:')
    success,result,my_history=agent_execute_with_retry(query,chat_history=my_history)
    print(result)
    my_history=my_history[-10:]