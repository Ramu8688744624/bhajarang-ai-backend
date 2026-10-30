import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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

# 1. Tool fetching category details to prevent confusing accessories with phones
@tool
def lookup_shop_catalog(search_query: str) -> str:
    """
    Search shop inventory for mobile accessories, covers, glasses, chargers, and parts.
    Pass extracted keywords such as brand, model name/number, or accessory type (e.g. 'Redmi', '6A', 'cover').
    """
    if not supabase:
        return "Database client is not initialized."

    try:
        tokens = [w.strip() for w in re.split(r'[\s,\-_?]+', search_query.lower()) if len(w.strip()) > 1]

        # Fetch active items along with category details
        response = (
            supabase.table("items")
            .select("item_name, model, selling_price, quantity, is_active, category_id, categories(name)")
            .eq("is_active", True)
            .execute()
        )

        rows = response.data
        if not rows:
            return "No inventory records found."

        matched_items = []
        for row in rows:
            cat_name = ""
            if row.get("categories") and isinstance(row["categories"], dict):
                cat_name = row["categories"].get("name", "")

            combined_text = f"{row.get('item_name', '')} {row.get('model', '')} {cat_name}".lower()

            if any(token in combined_text for token in tokens):
                matched_items.append({
                    "brand": row.get("item_name", ""),
                    "model": row.get("model", ""),
                    "category": cat_name if cat_name else "Accessory",
                    "price": row.get("selling_price", 0),
                    "stock": row.get("quantity", 0)
                })

        if not matched_items:
            return f"No catalog items matched query: '{search_query}'."

        results = []
        for it in matched_items:
            stock_str = f"{it['stock']} in stock" if it["stock"] > 0 else "Out of stock"
            results.append(
                f"- Category: {it['category']} | Model: {it['brand']} {it['model']} | Price: Rs.{it['price']} | Status: {stock_str}"
            )

        return "\n".join(results)

    except Exception as e:
        # Fallback if categories table join has a foreign key alias issue
        try:
            fallback_res = (
                supabase.table("items")
                .select("item_name, model, selling_price, quantity, is_active")
                .eq("is_active", True)
                .execute()
            )
            rows = fallback_res.data or []
            fallback_matches = []
            for r in rows:
                text = f"{r.get('item_name', '')} {r.get('model', '')}".lower()
                if any(t in text for t in tokens):
                    qty = r.get("quantity", 0)
                    stk = f"{qty} in stock" if qty > 0 else "Out of stock"
                    fallback_matches.append(f"- Item/Cover: {r.get('item_name')} {r.get('model')} | Price: Rs.{r.get('selling_price')} | Status: {stk}")
            return "\n".join(fallback_matches) if fallback_matches else f"No matches found for '{search_query}'."
        except Exception as inner_err:
            return f"Database search error: {str(inner_err)}"

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1
)
llm_with_tools = llm.bind_tools([lookup_shop_catalog])

SYSTEM_PROMPT = """
You are the virtual shop assistant for 'Jai Bhajarang Mobiles' located at Khaman to Bus Stand Road, Karimnagar.
We sell mobile accessories (back covers, tempered glass, chargers, earphones) and provide repair services. We DO NOT sell brand new mobile handsets unless explicitly stated.

CRITICAL WORKFLOW & CONVERSATION RULES:

1. UNDERSTAND THE INVENTORY:
   - When the catalog tool returns an item, pay close attention to the CATEGORY.
   - For example, if the result is "Category: Mobile Covers | Model: Redmi 6A | Price: Rs.80", that is a BACK COVER for the Redmi 6A, NOT the Redmi 6A smartphone! Never tell the user that the phone handset is Rs.80.

2. CONVERSATION STEP 1 (When user mentions just a phone model, e.g. "Do you have Redmi 6A?"):
   - Call the lookup tool for that model.
   - Look at what items exist in the catalog for that model.
   - Do NOT say "Yes, we have Redmi 6A in stock for Rs.80".
   - Instead, state what accessories are available for that model and ask what they need.
   - Example response:
     "For Redmi 6A, we currently have Mobile Covers available in stock. Are you looking for a back cover, tempered glass, or a repair service?"

3. CONVERSATION STEP 2 (When user specifies the item, e.g. "back cover" or "cover"):
   - Verify stock and give the exact price:
     "Yes! We have back covers for Redmi 6A in stock for Rs.80 (3 units available). You can visit our counter on Khaman to Bus Stand Road, Karimnagar to pick it up!"

4. LANGUAGE MATCHING:
   - User writes in English -> Reply in English.
   - User writes in Telugu script -> Reply in Telugu script.
   - User writes in Romanized Telugu/Tanglish (e.g. "Redmi 6A unaya?", "Cover rate entha?") -> Reply in natural Romanized Telugu (e.g. "Redmi 6A kosam maa daggara Mobile Covers unnaayi andi. Meeku cover kaavaala, tempered glass aa, leka screen repair aa?").

Keep answers polite, accurate, and concise.
"""

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