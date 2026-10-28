import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()

app = FastAPI(title="Jai Bhajarang Mobiles AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# Expanded Catalog including common accessories & repairs
INVENTORY_DATA = [
    {"name": "Redmi Note 13 Pro Tempered Glass", "category": "screen guard tempered glass protector", "price": 199, "stock": 15},
    {"name": "iPhone 15 Matte Screen Guard", "category": "screen guard tempered glass protector", "price": 299, "stock": 5},
    {"name": "Wired Earphones (3.5mm / Type-C)", "category": "earphones headphones audio", "price": 249, "stock": 12},
    {"name": "Wireless Bluetooth Earbuds / Neckband", "category": "earphones bluetooth neckband audio", "price": 899, "stock": 7},
    {"name": "Samsung 25W Type-C Original Adapter", "category": "charger adapter fast charger", "price": 1299, "stock": 8},
    {"name": "Boat Type-C Braided Cable 1m", "category": "cable data cable fast charging", "price": 249, "stock": 0},
    {"name": "Doorstep Screen / Display Replacement", "category": "service repair display screen break", "price": 1499, "stock": 999},
    {"name": "Battery Replacement Service", "category": "service repair battery charging issue", "price": 999, "stock": 999},
]

@tool
def lookup_shop_catalog(search_query: str) -> str:
    """
    Search shop catalog for accessories, audio, chargers, covers, and repair services.
    Pass translated English search terms (e.g. 'earphones', 'screen replacement', 'tempered glass').
    """
    words = [w.strip() for w in search_query.lower().replace("?", "").replace(",", "").split() if len(w.strip()) > 1]
    matches = []
    
    for item in INVENTORY_DATA:
        item_text = f"{item['name']} {item['category']}".lower()
        matched_words = [w for w in words if w in item_text or w.rstrip("s") in item_text]
        if len(matched_words) >= max(1, len(words) // 2):
            stock_label = f"{item['stock']} in stock" if item["stock"] > 0 else "Currently out of stock"
            matches.append(f"- {item['name']}: Rs.{item['price']} ({stock_label})")
            
    if not matches:
        return f"No direct catalog match found for '{search_query}'. Inform customer we can arrange it or invite them to visit our shop on Khaman to Bus Stand Road, Karimnagar."
        
    return "\n".join(matches)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2
)
llm_with_tools = llm.bind_tools([lookup_shop_catalog])

SYSTEM_PROMPT = (
    "You are the virtual assistant for 'Jai Bhajarang Mobiles' located at Khaman to Bus Stand Road, Karimnagar. "
    "We provide mobile accessories, screen repairs, battery replacement, and doorstep service in Karimnagar. "
    "LANGUAGE INSTRUCTIONS: "
    "1. When customers speak in Telugu or romanized Telugu (e.g. 'earphones unaya?', 'display change cheyali', 'cost entha?'), ALWAYS reply politely in friendly Telugu or conversational Telugu-English (e.g., 'Avunu andi, maa daggara earphones unnaayi...'). "
    "2. Before calling the lookup tool, translate the query keywords to English (e.g. 'unaya earphones' -> 'earphones'). "
    "3. Keep answers concise, polite, and helpful."
)

def process_message(user_text: str) -> str:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_text)
    ]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "lookup_shop_catalog":
                tool_result = lookup_shop_catalog.invoke(tool_call["args"])
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    )
                )
        
        final_response = llm_with_tools.invoke(messages)
        return final_response.content

    return response.content

@app.get("/")
def home():
    return {"status": "Online", "shop": "Jai Bhajarang Mobiles"}

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    bot_reply = process_message(request.message)
    return {"reply": bot_reply}