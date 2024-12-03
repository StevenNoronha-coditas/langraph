from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
load_dotenv()

class BlogGenerator(BaseModel):
    heading:str = Field(description = "Heading of Blog")
    content:str = Field(description = "Content of Blogs")

def get_llm():
    try:
        return ChatGroq(
            model_name="llama-3.1-70b-versatile"
            )
    except Exception as e:
        return "Exception occured"
    

def outline_generator(topic): 
    llm = get_llm()
    prompt = ChatPromptTemplate([
    ("system", "You are an expert AI bot, that based on the given topic generates a outline for the blog."),
    ("human", "Create a blog outline for the topic: {topic}")
    ])
    chain = prompt | llm
    response = chain.invoke({"topic": topic})
    # print(response.content)
    return response.content

def content_generator(outline):
    llm = get_llm()
    prompt = ChatPromptTemplate([
    ("system", "You are a helpful AI bot who is expert in creating content for blogs based on the given outline"),
    ("human", "Create content for the outline: {outline}")
    ])
    chain = prompt | llm
    response = chain.invoke({"outline": outline})
    # print(response.content)
    return response.content

def formatter(content):
    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=BlogGenerator)
    prompt = PromptTemplate(
    template="""You are a helpful AI bot who is expert in formatting the content generated. 
    Format the text into json object, use the given format instructions.
    Give the final output in a json object
    Content: {content}
    Format instructions: {format_instructions}""",
    )
    chain = prompt | llm | parser
    response = chain.invoke({"content": content, "format_instructions": parser.get_format_instructions()})
    print(response)
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate-blog', methods=['POST'])
def generate_blog():
    try:
        data = request.get_json()
        topic = data.get('topic')
        
        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
            
        outline = outline_generator(topic)
        content = content_generator(outline)
        formatted_content = formatter(content)
        
        # Ensure we're sending JSON response
        return jsonify({
            'success': True,
            'result': formatted_content
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add at the bottom of the file
if __name__ == '__main__':
    app.run(debug=True)