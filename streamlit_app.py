import streamlit as st
import json
import torch
from collections import Counter

import generate_text

#===========================================#
#        Loads Model and word_to_id         #
#===========================================#

with open('trained_model/word_to_id.json') as json_file:
    word_to_id = Counter(json.load(json_file))

with open('trained_model/always_capitalized.json') as json_file:
    always_capitalized = json.load(json_file)

id_to_word = ["<Unknown>"] + [word for word, index in word_to_id.items()]

net = torch.load('trained_model/trained_model.pt')
net.eval()




#===========================================#
#              Streamlit Code               #
#===========================================#
desc = "基于乐谱预训练的文本生成系统 "

st.title('基于乐谱预训练的文本生成系统')
st.write(desc)

user_input = st.text_input('Seed Text (can leave blank)')


if st.button('文本生成'):
    generated_text = generate_text.prediction(net, word_to_id, id_to_word, always_capitalized, user_input, 9, num_sentences)
    st.write(generated_text)
