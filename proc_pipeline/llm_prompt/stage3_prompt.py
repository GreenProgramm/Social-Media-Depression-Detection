content = stage2_output
stage3_prompt = f'''
You are a mental health expert. 

Given a structured description of a depression-related symptom, rewrite it into ONE natural, fluent paragraph that reflects how people actually express this feeling on social media. 

Requirements: 
1. Combine the symptom name, description, and expressions into a coherent paragraph. 
2. Use natural, informal language (similar to real posts). 
3. Include both explicit and implicit expressions. 
4. Avoid listing or bullet points. 
5. Do NOT repeat the original text verbatim. 
6. Make it suitable for semantic similarity matching. 

Input:
{content}
'''