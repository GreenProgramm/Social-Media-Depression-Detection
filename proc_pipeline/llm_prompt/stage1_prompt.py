cluster_output = cluster_data
query_group = []
for cluster_center in  cluster_output:
    stage1_prompt =f'''
    You are a clinical psychologist specializing in depression. 
    
    You are given a group of social media posts from users diagnosed with depression. These posts are semantically similar (clustered together). 
    
    Your task is NOT to summarize the posts, but to extract the underlying psychological symptom pattern. 
    
    Requirements: 
    1. Identify ONE core psychological symptom shared by these posts. 
    2. The symptom should be abstract (e.g., hopelessness, low motivation, self-blame). 
    3. Do NOT use generic labels like "negative emotion". 
    4. Capture implicit expressions, not only explicit ones. 
    5. Generate: 
       - A short symptom name 
       - A detailed description 
       - 5 representative expressions rewritten from the posts (generalized, not copied) 
       - 3-5 keywords 
       - An estimated severity level (low / medium / high) 
    
    Output strictly in JSON format: 
    {
    "symptom_name": "...", 
    "description": "...", 
    "representative_expressions": ["...", "...", "...", "...", "..."], 
    "keywords": ["...", "...", "..."], 
    "severity": "..." 
    } 
    Posts:
    {cluster_center}
    '''
    query_group.append(stage1_prompt)