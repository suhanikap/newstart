import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda

# -------------------------------
# Page
# -------------------------------
st.set_page_config(page_title="Idea Swarm Demo", layout="wide")
st.title("🐝 Swarm Pattern — Idea Generator (4 Peer Agents)")

st.markdown("""
**Swarm idea:** No central controller.  
Each agent works independently and writes to a **shared state (blackboard)**.  
Final output **emerges by combining** all contributions.
""")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Settings")

api_key = st.sidebar.text_input("Gemini API Key", type="password")

model = st.sidebar.selectbox(
    "Select Gemini Model",
    [
        "models/gemini-2.5-flash",
        "models/gemini-1.5-pro"
    ]
)

temperature = st.sidebar.slider("Creativity", 0.0, 1.0, 0.5)

# -------------------------------
# LLM
# -------------------------------
def get_llm():
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=model,
        temperature=temperature
    )

# -------------------------------
# PROMPTS (4 AGENTS)
# -------------------------------
business_prompt = PromptTemplate.from_template("""
You are a BUSINESS AGENT.

Generate monetization ideas for:
{topic}

Give concise bullet points.
""")

tech_prompt = PromptTemplate.from_template("""
You are a TECH AGENT.

Suggest product features / tech ideas for:
{topic}

Give concise bullet points.
""")

user_prompt_t = PromptTemplate.from_template("""
You are a USER AGENT.

List user pain points and needs for:
{topic}

Give concise bullet points.
""")

growth_prompt = PromptTemplate.from_template("""
You are a GROWTH AGENT.

Suggest distribution and growth strategies for:
{topic}

Give concise bullet points.
""")

# -------------------------------
# BUILD SWARM (Parallel)
# -------------------------------
def build_swarm(llm):
    business_chain = business_prompt | llm
    tech_chain = tech_prompt | llm
    user_chain = user_prompt_t | llm
    growth_chain = growth_prompt | llm

    swarm = RunnableParallel(
        business=business_chain,
        tech=tech_chain,
        user=user_chain,
        growth=growth_chain
    )

    return swarm

# -------------------------------
# AGGREGATION (Shared State View)
# -------------------------------
def aggregate_fn(outputs):
    return {
        "business": outputs["business"].content,
        "tech": outputs["tech"].content,
        "user": outputs["user"].content,
        "growth": outputs["growth"].content
    }

aggregate = RunnableLambda(aggregate_fn)

# -------------------------------
# UI
# -------------------------------
topic = st.text_area("Enter a problem / idea")

run = st.button("🚀 Run Swarm")

# -------------------------------
# MAIN
# -------------------------------
if run:

    if not api_key:
        st.error("Enter API key")
        st.stop()

    if not topic:
        st.error("Enter a topic")
        st.stop()

    llm = get_llm()
    swarm = build_swarm(llm)

    with st.spinner("Agents are thinking independently..."):
        outputs = swarm.invoke({"topic": topic})
        shared_state = aggregate.invoke(outputs)

    # -------------------------------
    # DISPLAY — AGENTS (PARALLEL VIEW)
    # -------------------------------
    st.subheader("🧠 Agent Outputs (Independent)")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.markdown("### 💰 Business Agent")
        st.write(shared_state["business"])

    with col2:
        st.markdown("### 🛠 Tech Agent")
        st.write(shared_state["tech"])

    with col3:
        st.markdown("### 👤 User Agent")
        st.write(shared_state["user"])

    with col4:
        st.markdown("### 📈 Growth Agent")
        st.write(shared_state["growth"])

    # -------------------------------
    # DISPLAY — SHARED STATE
    # -------------------------------
    st.markdown("---")
    st.subheader("🗂 Shared State (Blackboard)")

    st.json(shared_state)

    # -------------------------------
    # FINAL COMBINED VIEW
    # -------------------------------
    st.markdown("---")
    st.subheader("🎯 Combined Idea Pool")

    combined = "\n\n".join(shared_state.values())
    st.success(combined)
