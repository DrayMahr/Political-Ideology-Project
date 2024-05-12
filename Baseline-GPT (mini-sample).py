for i in postlist:
    jsondata=openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a moderate, pattern-following assistant."},
            {"role": "user", "content": "Judge whether the statement is liberal or conservative： Virtually nobody will be going over the wall and the few who manage it can be detected with various means of surveillance.  A real wall on the border will lessen, not increase the necessary manpower. If you think it's extremely liberal output 1. If you think it's normally liberal output 2. If you think it's slightly liberal output 3. If you think it's moderate output 4. If you think it's slightly conservative output 5. If you think it's normally conservative output 6. If you think it's extremely conservative output 7."},
            {"role": "assistant", "content": "6"},
            {"role": "user", "content": "Judge whether the statement is liberal or conservative：  the first three therapists I saw were trying to push hrt after first visit.Click to expand...That is ethically conscionable. HRT should be 90 days to even possibly six months in the future after psychological screening and a firm diagnosis is conducted.  There is no way that they can confirm that someone is transgendered after 45-50 minutes. If you think it's extremely liberal output 1. If you think it's normally liberal output 2. If you think it's slightly liberal output 3. If you think it's moderate output 4. If you think it's slightly conservative output 5. If you think it's normally conservative output 6. If you think it's extremely conservative output 7."},
            {"role": "assistant", "content": "1"},
            {"role": "user", "content": f"Judge whether the statement is liberal or conservative： {i} .If you think it's extremely liberal output 1. If you think it's normally liberal output 2. If you think it's slightly liberal output 3. If you think it's moderate output 4. If you think it's slightly conservative output 5. If you think it's normally conservative output 6. If you think it's extremely conservative output 7.just give me the number"}
        ]
    )
    json_data = str(jsondata)
    data = json.loads(json_data)
    content = data['choices'][0]['message']['content']
    labellist.append(content)

  for i in postlist:
      jsondata = openai.ChatCompletion.create(
          model="gpt-4",
          messages=[
              {"role": "system", "content": "You are a moderate, pattern-following assistant."},
              {"role": "user",
               "content": "Judge whether the statement is liberal or conservative： Virtually nobody will be going over the wall and the few who manage it can be detected with various means of surveillance.  A real wall on the border will lessen, not increase the necessary manpower. If you think it's extremely liberal output 1. If you think it's normally liberal output 2. If you think it's slightly liberal output 3. If you think it's moderate output 4. If you think it's slightly conservative output 5. If you think it's normally conservative output 6. If you think it's extremely conservative output 7."},
              {"role": "assistant", "content": "6"},
              {"role": "user",
               "content": "Judge whether the statement is liberal or conservative：  the first three therapists I saw were trying to push hrt after first visit.Click to expand...That is ethically conscionable. HRT should be 90 days to even possibly six months in the future after psychological screening and a firm diagnosis is conducted.  There is no way that they can confirm that someone is transgendered after 45-50 minutes. If you think it's extremely liberal output 1. If you think it's normally liberal output 2. If you think it's slightly liberal output 3. If you think it's moderate output 4. If you think it's slightly conservative output 5. If you think it's normally conservative output 6. If you think it's extremely conservative output 7."},
              {"role": "assistant", "content": "1"},
              {"role": "user",
               "content": f"Judge whether the statement is liberal or conservative： {i} .If you think it's extremely liberal output 1. If you think it's normally liberal output 2. If you think it's slightly liberal output 3. If you think it's moderate output 4. If you think it's slightly conservative output 5. If you think it's normally conservative output 6. If you think it's extremely conservative output 7.just give me the number"}
          ]
      )
      json_data = str(jsondata)
      data = json.loads(json_data)
      content = data['choices'][0]['message']['content']
      labellist1.append(content)