import requests
import re
import time
from graphviz import Source
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# Suppress optional warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
GROQ_MODEL = "llama3-70b-8192"

# ------------------------------------------------------------
# Call Groq LLM API to generate DOT code
# ------------------------------------------------------------
def query_groq(prompt: str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "user",
                # "content": f"Generate a Graphviz DOT diagram for the concept: {prompt}. Only return a markdown code block with the diagram using ```dot ...``` syntax."
                "content": f"""
                    Generate only a Graphviz DOT diagram in a markdown code block (```dot ... ```), for the concept: {prompt}
                    Only respond with a markdown code block like:
                    
                    ```dot
                    digraph concept {{
                        A -> B;
                    }}```
                    All the nodes should be connected in correct order using your full potential
                    It should be logically accurate using your full potential
                    If required for better explanationsin some case add arrow labels as well 
                    Do NOT include any explanation, text, or additional formatting.
                    Only return a single, clean DOT code block.
                    Only return ONE digraph block.
                    Do NOT use the word 'dot' or any code fencing artifacts outside the code block. 
                    """
            }
        ],
        "temperature": 0.7,
        "stream": False
    }

    print("⏳ Sending request to Groq LLM...")
    try:
        start = time.time()
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        elapsed = time.time() - start
        print(f"✅ Groq responded in {elapsed:.2f} seconds")
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as http_err:
        print(f"❌ Groq API error: {http_err}")
        if http_err.response is not None:
            try:
                print("📄 Groq response JSON:", http_err.response.json())
            except Exception:
                print("📄 Raw response:", http_err.response.text)
        return None
    except Exception as e:
        print(f"❌ Other error: {e}")
        return None
 
        # print("📄 Response (if available):", getattr(e, 'response', None))
        # return None


# ------------------------------------------------------------
# Extract DOT code from model response
# ------------------------------------------------------------
def extract_dot_code(response):
    # 1. Try to extract from ```dot ... ```
    match = re.search(r"```dot\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. Try to extract raw digraph or graph {...}
    match = re.search(r"(digraph|graph)\s+\w*\s*{.*?}", response, re.DOTALL)
    if match:
        return match.group(0).strip()

    print("❌ DOT code block not found. Full response:")
    print(response)
    return None


# ------------------------------------------------------------
# Fix and wrap DOT code
# ------------------------------------------------------------
def fix_dot_syntax(dot_code: str) -> str:
    # 🔧 Remove lines that open or close a nested digraph or graph
    cleaned_lines = []
    for line in dot_code.splitlines():
        line = line.strip()
        if line.lower().startswith("digraph ") or line.lower().startswith("graph "):
            print(f"⚠️ Removing nested block line: {line}")
            continue
        if line == "}":
            print(f"⚠️ Removing stray closing brace: {line}")
            continue
        cleaned_lines.append(line)

    dot_code = "\n".join(cleaned_lines)

    # Proceed with rest of syntax fix
    dot_code = dot_code.replace('-->', '->')
    node_labels = {}
    edges = set()
    fixed_lines = []

    for line in dot_code.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue

        pattern = re.findall(r'(\w+)\[(.*?)\]', line)
        if pattern:
            for node, label in pattern:
                node_labels[node.strip()] = label.strip()
            clean_line = re.sub(r'(\w+)\[.*?\]', r'\1', line)
            if '->' in clean_line:
                edges.add(clean_line.strip(';') + ';')
            continue

        match = re.match(r'^(\w+)\s*\[(.+?)\];?$', line)
        if match:
            node, label = match.groups()
            node_labels[node.strip()] = label.strip()
            continue

        if re.match(r'^\w+ -> \w+;?$', line):
            edges.add(line.strip(';') + ';')
            continue

        print(f"⚠️ Unrecognized DOT line: {line}")
        fixed_lines.append(line)

    # for node, label in sorted(node_labels.items()):
    #     fixed_lines.append(f'{node} [label="{label}"];')'

    for node, label in sorted(node_labels.items()):
        # Remove nested label= if it already exists
        if label.lower().startswith("label="):
            label = label.split("=", 1)[1].strip().strip('"')
        fixed_lines.append(f'{node} [label="{label}"];')


    fixed_lines.extend(sorted(edges))

    return "\n".join(["digraph concept_graph {", "rankdir=TB;"] + fixed_lines + ["}"])

# ------------------------------------------------------------
# Render DOT code using Graphviz
# ------------------------------------------------------------
# def render_dot(dot_code, filename="concept_graph"):
#     try:
#         s = Source(dot_code, filename=filename, format="png")
#         output_path = s.render(cleanup=True)
#         print(f"📌 Diagram saved to: {output_path}")
#         img = Image.open(output_path)
#         plt.imshow(img)
#         plt.axis("off")
#         plt.title("Generated Diagram")
#         plt.show()
#     except Exception as e:
#         print("❌ Failed to render diagram:", e)

# # ------------------------------------------------------------
# # Entry point
# # ------------------------------------------------------------
# def generate_diagram_from_groq(prompt):
#     response = query_groq(prompt)
#     if not response:
#         return

#     dot_code = extract_dot_code(response)
#     if not dot_code:
#         print("❌ Could not extract DOT code.")
#         # print("📄 Full response:", response)
#         return

#     dot_code = fix_dot_syntax(dot_code)
#     print("\n🔧 Final DOT Code:\n", dot_code)
#     render_dot(dot_code)


# 🔄 NEW Streamlit-compatible version
def generate_diagram_streamlit(concept):
    response = query_groq(concept)
    if not response:
        st.error("❌ Failed to get response from LLM.")
        return

    dot_code = extract_dot_code(response)
    if not dot_code:
        st.error("❌ Failed to extract diagram DOT code.")
        st.text(response)
        return

    fixed_code = fix_dot_syntax(dot_code)

    try:
        s = Source(fixed_code, format="png")
        output_path = s.render(cleanup=True)
        # st.image(output_path, caption="📘 Concept Diagram", use_container_width=True)
        return output_path  # ⬅️ Return file path instead of rendering
    except Exception as e:
        st.error(f"❌ Diagram rendering failed: {e}")
        return

