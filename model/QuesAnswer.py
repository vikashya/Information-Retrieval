import string
import re
import os
import numpy as np
import pickle
import math
import tensorflow as tf
import tensorflow_hub as hub
from elasticsearch import Elasticsearch

class InfoRetrieval:

    def __init__(self,module_url="https://tfhub.dev/google/universal-sentence-encoder/4"):
        self.datadir = 'data/'
        self.batchsize= 2000
        self.sentembedfile=''
        self.corpus =[]
        self.module_url = module_url
        self.model = hub.load( module_url )
        print('Inside constructor : ',__name__)

    def gen_clean_data(self,filename):
        cfilename = os.path.splitext(filename)[0]+'-cleansed'+os.path.splitext(filename)[1]
        with open( self.datadir+cfilename, 'w+' ) as out_file:
            with open( self.datadir + filename, 'r' ) as in_file:
                for sent in in_file:
                    sent = sent.rstrip('\n')
                    out_file.write( self.clean_sent( sent )+'\n')
        return cfilename

    def clean_sent(self ,s):
        s_ = s.lower()
        table = str.maketrans( '', '', string.punctuation )
        #s_= [w.translate(table) for w in s_]
        s_ = re.sub(r"i'm", "i am", s_)
        s_ = re.sub( r"it's", "it is", s_ )
        s_ = re.sub(r"he's", "he is", s_)
        s_ = re.sub(r"she's", "she is", s_)
        s_ = re.sub(r"that's", "that is", s_)
        s_ = re.sub(r"what's", "what is", s_)
        s_ = re.sub(r"where's", "where is", s_)
        s_ = re.sub(r"\'ll", " will", s_)
        s_ = re.sub(r"\'ve", " have", s_)
        s_ = re.sub(r"\'re", " are", s_)
        s_ = re.sub(r"\'d", " would", s_)
        s_ = re.sub(r"won't", "will not", s_)
        s_ = re.sub(r"can't", "cannot", s_)
        s_ = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", " ", s_)
        s_ = re.sub( r'[^\x00-\x7F]+', ' ', s_ )
        s_ = re.sub( r'[(\n+)+]', ' ', s_ )
        s_.encode( 'ascii', errors='ignore' ).strip().decode('ascii')
        return s_

    def embed(self,input):
        return self.model(input)

    def create_corpus_embedding(self,fname,batchsize=2000):
        print ('Creating Embedding for Filename: ',fname)
        embed_list = []
        self.embedfile = os.path.splitext( fname )[0] + '-embed'
        self.sentembedfile =self.datadir+self.embedfile
        with open(self.datadir+fname, 'r') as file:
            self.corpus = [line.rstrip('\n') for line in file]
            batch_ctr = 1
            if( batchsize > len(self.corpus)):
                batchsize = len(self.corpus)
            print ('Running for BatchSize',batchsize)
            for batch in np.array_split( self.corpus, batchsize ):
                print( 'Running Batch : ', batch_ctr )
                embed_list.extend(self.embed( batch ) )
                batch_ctr += 1
        with open(self.sentembedfile, 'wb') as handle:
            pickle.dump( embed_list,handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Embedding created in Filename: ', self.embedfile)


    def find_similar_answer_elastic(self,ques ,match):
        elastic = Elasticsearch( [{'host': 'localhost', 'port': 9200}])
        res = elastic.search(index='cb', doc_type='medicalqa', body={
            'query': {
                'match': {
                    "answer": ques
                }
            }
        })
        tot_res = res['hits']['hits'][0:match+1]
        top_res= ['Score: '+ str(r['_score']) + ': '+ str(r['_source']['answer']) for r in tot_res]
        return top_res

    def find_similar_answer_uce(self,ques ,fname,match,genEmbedding=False):
        print( 'Inside Search for:',ques )
        quesembed = self.embed([ques])
        print( 'Generated Embedding for Question' )
        if (genEmbedding):
            self.create_corpus_embedding(fname, batchsize=2000 )
            print( 'Re-Generated Embedding for File' )
        if (self.sentembedfile==''):
            print('Inside Corpus Loading')
            self.sentembedfile = self.datadir + os.path.splitext( fname )[0] + '-embed'
            with open( self.datadir + fname, 'r' ) as file:
                self.corpus = [line.rstrip( '\n' ) for line in file]
        with open( self.sentembedfile, 'rb' ) as handle:
            sentembed = pickle.load( handle )
        print( 'Loaded Sentence Embedding for File' )
        similar_sent_matches = {}
        sts_encode1 = tf.nn.l2_normalize(quesembed, axis=1)
        sts_encode2 = tf.nn.l2_normalize(sentembed, axis=1)
        cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
        scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
        scores_list = scores.numpy().tolist()
        top_matches = sorted( range(len(scores_list) ), key=lambda i: scores_list[i], reverse=True )[:match]
        for top_match in top_matches:
            similar_sent_matches['idx:'+str(top_match)+':Score:'+str(scores_list[top_match])]=self.corpus[top_match]
        return (similar_sent_matches)
