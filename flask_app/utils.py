import os
import requests
from bs4 import BeautifulSoup
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import re
from dotenv import load_dotenv
import random

# Load API keys from environment variables
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def refine_query(raw_query):
    if not GEMINI_API_KEY:
        return raw_query  # fallback to original query
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2
    )
    prompt = f"Rewrite this user query to optimize for best Google search results: '{raw_query}'"
    response = llm.invoke(prompt)
    return response.content.strip()

def search_articles_serper(query):
    if not SERPER_API_KEY:
        print("Warning: SERPER_API_KEY not found, skipping Serper search.")
        return []

    search_url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query}

    try:
        response = requests.post(search_url, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()

        articles = []
        for result in results.get("organic", []):
            title = result.get("title", "No title")
            link = result.get("link", "No link")
            snippet = result.get("snippet", "No snippet")
            articles.append({"title": title, "link": link, "snippet": snippet})
        return articles

    except requests.exceptions.RequestException as e:
        print(f"Error fetching search results from Serper: {e}")
        return []

def search_articles_duckduckgo(query):
    print(f"Trying DuckDuckGo search for: {query}")
    url = f"https://html.duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        articles = []
        results = soup.find_all('a', class_='result__a', limit=5)  # Limit to top 5 results
        for result in results:
            title = result.get_text()
            link = result.get('href')
            articles.append({"title": title, "link": link, "snippet": ""})
        return articles

    except Exception as e:
        print(f"Error fetching from DuckDuckGo: {e}")
        return []

def search_articles(query):
    articles = search_articles_serper(query)
    if not articles:
        print("No results from Serper, falling back to DuckDuckGo...")
        articles = search_articles_duckduckgo(query)
    return articles

def fetch_article_content(url):
    # Skip YouTube or unsupported domains
    if "youtube.com" in url or "youtu.be" in url:
        print(f"Skipping YouTube link: {url}")
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if res.status_code == 403:
            print(f"403 Forbidden: Skipping {url}")
            return ""
        print(f"HTTP Error fetching article from {url}: {e}")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"Error fetching article from {url}: {e}")
        return ""

    soup = BeautifulSoup(res.text, "html.parser")
    headings = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])]
    paragraphs = [p.get_text() for p in soup.find_all('p')]

    if not headings and not paragraphs:
        print(f"Warning: No content found for {url}")
        return ""

    content = "\n".join(headings + paragraphs)
    return content.strip()

def concatenate_content(articles):
    full_text = ""
    for article in articles:
        url = article.get("link")
        title = article.get("title")
        content = fetch_article_content(url)
        if content:
            full_text += f"\n\nTitle: {title}\n"
            full_text += f"URL: {url}\n"
            full_text += f"Content: {content}\n"
    return full_text.strip()

def generate_answer(content, query):
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found.")
        return "API Key missing."

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.4
    )

    if not content:
        print("No content available. Searching online directly...")
        articles = search_articles(query)
        full_text = concatenate_content(articles)

        if full_text:
            documents = [Document(page_content=full_text)]
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GEMINI_API_KEY
            )
            vectorstore = FAISS.from_documents(documents, embeddings)
            retriever = vectorstore.as_retriever()

            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=False
            )
            result = chain.invoke({"question": query})
            return result["answer"].strip()
        else:
            response = llm.invoke(query)
            return response.content.strip()

    documents = [Document(page_content=content)]
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )

    result = chain.invoke({"question": query})
    return result["answer"].strip()
