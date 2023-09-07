from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback
from googletrans import Translator


def Answering(document,question):
    OPENAI_API_KEY='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'
    translator = Translator()

    template = """
    Answer the question based on EXAMPLE1 and EXAMPLE2 data, the question is {question}, answer in a simple way, not too long.
    % START OF EXAMPLE1
    {example1}
    % END OF EXAMPLE1
    % START OF EXAMPLE2
    {example2}
    % END OF EXAMPLE2

    YOUR ANSWER:
    """
    example1 = document['matches'][0]['metadata']['Description']
    example2 = document['matches'][1]['metadata']['Description']
    prompt = PromptTemplate(
        input_variables=["example1","example2",'question'],
        template=template,
    )
    final_prompt = prompt.format(example1=example1,example2=example2,question=question.text)

    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo')
    with get_openai_callback() as cb:
        output = llm.predict(final_prompt)
        print(output)
        print(cb)
    output = translator.translate(output, to_lang='th',dest='th')
    print(output.text)
    return output.text

