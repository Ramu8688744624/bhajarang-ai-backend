import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# 1. Load environment secrets
load_dotenv()

# 2. Initialize FastAPI App
app = FastAPI(title="Jai Bhajarang Mobiles AI Assistant")

# Enable CORS so your website can talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend (like Vercel)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Request Data Schema
class ChatRequest(BaseModel):
    message: str

# 4. Mock Inventory Data
INVENTORY_DATA = [
    {"name": "Redmi Note 13 Pro Tempered Glass", "category": "screen_guard", "price": 199, "stock": 15},
    {"name": "iPhone 15 Matte Screen Guard", "category": "screen_guard", "price": 299, "stock": 5},
    {"name": "Samsung 25W Type-C Original Adapter", "category": "charger", "price": 1299, "stock": 8},
    {"name": "Boat Type-C Braided Cable 1m", "category": "cable", "price": 249, "stock": 0},
    {"name": "Doorstep Screen Repair Service", "category": "service", "price": 1499, "stock": 999},
]

# 5. Inventory Tool
# 5. Inventory Tool (Updated to handle partial words and plural terms)
@tool
def lookup_shop_catalog(search_query: str) -> str:
    """
    Search the shop catalog and inventory for accessories, services, prices, and stock levels.
    Always trigger this tool when a user asks about price, availability, or repair services.
    """
    # Clean query into individual search tokens (e.g., 'iphone', '15', 'screen')
    words = [w.strip() for w in search_query.lower().replace("?", "").replace(",", "").split() if len(w.strip()) > 1]
    matches = []
    
    for item in INVENTORY_DATA:
        item_text = f"{item['name']} {item['category']}".lower()
        # If at least half the customer's search words appear in the product name
        matched_words = [w for w in words if w in item_text or w.rstrip("s") in item_text]
        if len(matched_words) >= max(1, len(words) // 2):
            stock_label = f"{item['stock']} units available" if item["stock"] > 0 else "Currently out of stock"
            matches.append(f"- {item['name']}: Rs.{item['price']} ({stock_label})")
            
    if not matches:
        return f"No direct catalog match found for '{search_query}'. Inform customer to visit shop counter."
        
    return "\n".join(matches)
# 6. Initialize LLM with Tools
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1
)
llm_with_tools = llm.bind_tools([lookup_shop_catalog])

# 7. Core Query Handler
def process_message(user_text: str) -> str:
    messages = [
        SystemMessage(
            content=(
                "You are the virtual shop assistant for 'Jai Bhajarang Mobiles' located in Karimnagar. "
                "Respond helpfully, politely, and concisely. "
                "Always check product availability and pricing through your lookup tool before confirming to the customer. "
                "You understand English, Telugu, and Hinglish queries seamlessly."
            )
        ),
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

# 8. Web API Endpoints
@app.get("/")
def home():
    return {"status": "Online", "shop": "Jai Bhajarang Mobiles AI Assistant"}

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    bot_reply = process_message(request.message)
    return {"reply": bot_reply}