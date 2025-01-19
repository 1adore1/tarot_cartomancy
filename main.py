import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
import cv2

API_KEY = st.secrets["API_KEY"]
client = InferenceClient(api_key=API_KEY)

df = pd.read_csv('data/tarot_short.csv')
card_to_meaning = dict(zip(df['name'], df['short_meaning']))
card_to_img = dict(zip(df['name'], df['img']))

def get_triplet():
    return np.random.choice(list(card_to_meaning.keys()), 3, replace=False)

def display_cards(cards):
    imgs = []
    for card in cards:
        imgs.append(cv2.imread('cards/' + card_to_img[card]))
    for i, col in enumerate(st.columns(3)):
        with col:
            st.image(imgs[i], caption=cards[i])

def interpret_cards(question, cards):
    prompt = f"""
    Question: {question}
    Cards drawn: {', '.join(cards)}
    Card meanings:
    - {cards[0]}: {card_to_meaning[cards[0]]}
    - {cards[1]}: {card_to_meaning[cards[1]]}
    - {cards[2]}: {card_to_meaning[cards[2]]}
    Write a structured response in the question's language:  
    - First three paragraphs (2 sentences each) interpret each card in order.  
    - The last paragraph (2 sentences) summarizes and gives a final answer. 
    """
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct", 
        messages=messages, 
        max_tokens=350, 
        temperature=0.7
    )

    return completion.choices[0].message

def main():
    st.title("Tarot Card Reader")
    question = st.text_input("Ask your question: ")
    if question:
        cards = get_triplet()
        display_cards(cards)
        answer = interpret_cards(question, cards)
        st.write(answer['content'])

if __name__ == "__main__":
    main()