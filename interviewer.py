from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import random

with open("data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Remove empty lines and strip whitespace
questions = [line.strip() for line in lines if line.strip()]


selected_questions = random.sample(questions,3)
user_answers = []
print("Answer the following questions:\n")

for i, question in enumerate(selected_questions, 1):
    print(f"Q{i}: {question}")
    ans = input("Your answer: ")
    user_answers.append((question, ans))

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    api_key="AIzaSyA3zAmVqDYHz_qlF5EOuUX0vYTgyq9cco0"
)
def grade_answer(question, answer):
    messages = [
        SystemMessage(content="You are a strict evaluator for data structure questions. Grade the user's answer from 0 to 10 and give brief feedback."),
        HumanMessage(content=f"Question: {question}\nUser Answer: {answer}\nGive score and feedback.")
    ]
    response = llm.invoke(messages)  
    return response

for i, (q, a) in enumerate(user_answers, 1):
    result = grade_answer(q, a)
    print(f"\nEvaluation for Q{i}:\n{result}")