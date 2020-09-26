from QuesAnswer import InfoRetrieval
import argparse

# Name of corpus data file
datafile = "ques-ans-dataset.txt"

# Build the model
e1 = InfoRetrieval(module_url="https://tfhub.dev/google/universal-sentence-encoder/4")
cfname = e1.gen_clean_data(datafile)
e1.create_corpus_embedding(fname=cfname,batchsize=100)


# Get the name of Question
parser = argparse.ArgumentParser()
parser.add_argument("-ques", help="this is my question")
args = parser.parse_args()
ques = args.q
ques= ques.replace('?','')
ques= ques.replace('.','')
print("Question is : ",ques)

# Find similar match
if (len(ques.split() > 3)):
    ans = e1.find_similar_answer_uce(ques='How are you?',fname=cfname,match=3,genEmbedding=False)
else:
    ans = e1.find_similar_answer_elastic(ques='How are you?', match=3)

print(ans)