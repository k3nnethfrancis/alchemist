import lilypad; lilypad.configure()
from openai import OpenAI
 
client = OpenAI()
 
@lilypad.generation()
def answer_question(question: str) -> str:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    return str(completion.choices[0].message.content)
 
if __name__ == "__main__":
    lilypad.configure()
    answer = answer_question("What is the meaning of life?")
    print(answer)