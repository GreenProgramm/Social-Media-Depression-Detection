content = stage1_output
stage2_prompt = f'''
You are refining a psychological symptom scale. 

Given a list of extracted symptoms from different clusters: 

Tasks: 
1. Merge similar symptoms 
2. Split overly broad ones if needed 
3. Ensure each symptom is distinct and clinically meaningful 
4. Normalize naming style 
5. Do not retain symptom names that are duplicates, overly broad, or merely express general emotions. 

Please output strictly in JSON list format: 
[
{content}
]
'''