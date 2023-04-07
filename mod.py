from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import json
from datasets import Dataset
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import math
import collections

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M", bos_token='<|startoftext|>', eos_token='<|endoftext|>')
tokenizer.pad_token = tokenizer.eos_token

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

model.load_state_dict(torch.load('./gpt_last_ft.pt', map_location=torch.device('cpu'))) # 여기서는 dict만 불러온다. 애초에 dict만 저장함.
model.eval()

def generate(input, info=""):
  # model에 넣어서 response 생성
  
  found=find_informations(input)
  informations=""
  for x in list(found[0].values())[:2]:
    print("-=-ef=w-w=-=", x)
    if x[0]>76:
      informations += x[1] + " "
  informations += json_data[0]
  print(json_data[0], "이거 나오나?")
  
  prompt=f"Human:{input}\n\nInformation:{informations}\n\nAssistant:"
  tokenized = tokenizer(prompt, return_tensors="pt")
  input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']

  generated = model.generate(
      input_ids, max_length=1024, num_beams=5, no_repeat_ngram_size=3, early_stopping=True, temperature=0.3,
  )
  return tokenizer.decode(generated[0], skip_special_tokens=True)

torch.set_grad_enabled(False) # https://runebook.dev/ko/docs/pytorch/generated/torch.set_grad_enabled
bert_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
bert_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

with open('./harry.json') as json_file:
    json_data = json.load(json_file)

HarryDict = dict()
HarryDict['id']=[x for x in range(len(json_data))]
HarryDict['sent']=[x for x in json_data]

dataset = Dataset.from_dict(HarryDict)

embeddings1 = dataset.map(lambda x: {'embeddings': bert_model(**bert_tokenizer(x["sent"],padding=True, truncation=True, return_tensors='pt'))[0][0].numpy()})
embeddings1.add_faiss_index(column='embeddings')

num_of_sentences=30

def find_informations(query):
  def get_tf_results(queries):
    queries = [preprocessing_query(x) for x in queries]
    tf_weight_list = [x for x in range(50, 0, -1)]
    data = queries + json_data

    tfidf_vect = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vect.fit_transform(data)
    cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    tf_result = []
    for idx in range(len(queries)):
      injected_list=[]
      sim_scores = [(i, c) for i, c in enumerate(cosine_matrix[idx]) if i != idx]
      sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse=True)
      sim_scores = sim_scores[:num_of_sentences]
      i=0
      for (index, score) in sim_scores:
        if sim_scores[i][1]==0:
          cal_score=0
        else:
          cal_score=(sim_scores[i][1]/3+1.2)*tf_weight_list[i]
        injected_list.append({"index":index-len(queries), "score":cal_score, "sent":data[int(sim_scores[i][0])]})
        i+=1
      # for i in range(num_of_sentences):
      #     print(f'score : {injected_list[i]["score"]}, Found : {injected_list[i]["sent"]}')
      tf_result.append(injected_list)
    return tf_result

  def get_sim_results(queries):
    sim_results=[]
    k=num_of_sentences

    for orig_query in queries:
      query = preprocessing_query(orig_query)
      question_embedding = q_encoder(**q_tokenizer(query, return_tensors="pt"))[0][0].detach().numpy()
      scores, retrieved_examples = embeddings1.get_nearest_examples('embeddings', question_embedding, k=k)
      df_dpr = pd.DataFrame.from_dict(retrieved_examples)
      df_dpr["scores"] = scores
      df_dpr.sort_values("scores", ascending=False, inplace=True)
      sim_results.append({
          "query":orig_query,
          "result":df_dpr[['id', 'sent', 'scores']]
      })
    return sim_results
  
  queries=[query]
  sim_results = get_sim_results(queries)
  queries = [preprocessing_query(x) for x in queries]
  tf_results=get_tf_results(queries)

  all_results=[]
  hyper = 0.25

  for (idx, sim) in enumerate(sim_results):
    print("=-------------------"*4)
    result_dict=collections.defaultdict(int)
    for i in range(len(sim['result'])):
      result_dict[sim['result'].iloc[i]['id']]=[sim['result'].iloc[i]['scores']*(1-hyper), sim['result'].iloc[i]['sent']]
    for tf in tf_results[idx]:
      if result_dict[tf['index']]:
        print("겹침", math.floor(result_dict[tf['index']][0]*100)/100, math.floor(tf['score']*100)/100, tf['sent'])
        result_dict[tf['index']] = [tf['score']*hyper + result_dict[tf['index']][0], result_dict[tf['index']][1]]
      else:
        result_dict[tf['index']]=[tf['score']*(hyper), tf['sent']]
    sorted_dict = dict(sorted(result_dict.items(), key=lambda item: item[1],reverse=True))
    all_results.append(sorted_dict)
  
  return all_results

def preprocessing_query(data, plus=True):
  if "yours" in data:
    data=data.replace('yours', 'My')
  if 'your' in data:
    data=data.replace('your', 'My')
  if 'you' in data:
    data=data.replace('you', 'I')
  if plus == True:  
    if "friend" in data:
      data += " friends"
    if "sport" in data:
      data += " sports"
  
  data=data.replace('are', 'am')
  
  return data