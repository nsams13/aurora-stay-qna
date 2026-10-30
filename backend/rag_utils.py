
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    rules = """
    1. Use only information explicitly stated in the retrieved policy information. Do not use outside knowledge or assumptions.
    2. Answer the question that was asked, directly, in the first sentence.
    3. Quote exact figures, times and fees as written (for example dollar amounts, 3:00 PM, $150 stay cap) and name the property or rate plan they belong to. Never transfer a fee from one property to another.
    4. Include the conditions, exceptions and required checks that change the answer (for example the confirmation controls, manager approval, availability, live property profile).
    5. If the question depends on a property, rate plan or booking channel that is not given or not covered in the retrieved text, say what it depends on instead of guessing.
    6. If the retrieved information does not contain the answer, say: "The available policy documents do not contain enough information to answer this question." Then say what staff should check (for example the live property profile or the property team).
    7. If retrieved passages conflict, point out the conflict.
    8. Do not include unrelated information. Keep the answer short, clear and written for hotel staff.
    9. Do not promise refunds, fee waivers or exceptions that the policy leaves to a manager.
    """

    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    {rules}

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
