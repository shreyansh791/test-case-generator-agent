# QA Test Case Generator - Part 2: Streamlit UI Version (Agent with Tools)
#
# Flow: Streamlit UI -> User Input -> Prompt -> Agent -> (Tools) -> LLM -> Test Cases -> UI
#
# Same application as Part 1, but the input() calls are replaced with
# Streamlit widgets and the test cases are shown on a web page.
#
# Setup:
#   pip install langchain langchain-openai python-dotenv streamlit
#   Create a .env file containing: OPENAI_API_KEY=your_key_here
# Run it with: streamlit run qa_testcase_generator_app.py

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

# Step 1: Load the API key from the .env file.
load_dotenv()

# Change this if you want to use a different model.
MODEL = "openai:gpt-5.5"

TEST_TYPES = ["Functional", "Positive", "Negative", "Boundary", "All"]


# ---------------------------------------------------------------------------
# Step 2: Tools (same as Part 1)
# The @tool decorator turns a normal Python function into a tool the agent can
# call by itself. The docstring tells the agent WHAT the tool does and WHEN to use it.
# ---------------------------------------------------------------------------

@tool
def get_test_type_guidelines(test_type: str) -> str:
    """Get the testing guidelines for a test type.
    Call this FIRST, before writing any test cases.
    test_type must be one of: Functional, Positive, Negative, Boundary, All."""
    guidelines = {
        "Functional": (
            "Functional testing: verify each feature works exactly as described in the "
            "requirement. Cover every business rule and main user flow."
        ),
        "Positive": (
            "Positive testing: use valid inputs and expected user actions. "
            "Confirm the system accepts them and gives the correct result."
        ),
        "Negative": (
            "Negative testing: use invalid inputs, empty fields, wrong formats, and "
            "unexpected actions. Confirm the system rejects them and shows clear error messages."
        ),
        "Boundary": (
            "Boundary testing: test values at the limits: minimum, maximum, just below "
            "the minimum, and just above the maximum (lengths, numbers, sizes, dates)."
        ),
    }

    test_type = test_type.strip().capitalize()

    if test_type == "All":
        return "Create a balanced mix of these types:\n" + "\n".join(guidelines.values())

    if test_type in guidelines:
        return guidelines[test_type]

    return f"Unknown test type '{test_type}'. Use Functional, Positive, Negative, Boundary, or All."


@tool
def generate_test_case_ids(count: int) -> list[str]:
    """Generate the test case IDs to use, in order (TC_001, TC_002, ...).
    Call this before writing the test cases, and use exactly these IDs,
    one per test case, so the number of test cases is always correct."""
    ids = []
    for number in range(1, count + 1):
        ids.append(f"TC_{number:03d}")
    return ids


@tool
def get_priority_guidelines() -> str:
    """Get the rules for assigning High, Medium, or Low priority to a test case.
    Call this before assigning priorities."""
    return (
        "High: core functionality, security, payments, data loss, or anything that "
        "blocks the user from completing the main task.\n"
        "Medium: important but non-blocking features, validations, and error messages.\n"
        "Low: cosmetic issues, rare edge cases, and minor usability details."
    )


# ---------------------------------------------------------------------------
# Step 3: System prompt
# Same as Part 1, except the output format is now Markdown, because Streamlit
# displays Markdown nicely (headings, bold labels, numbered steps).
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a Senior QA Engineer with over 10 years of experience in manual and
automation testing. You write clear, precise, and professional test cases
from software requirements.

Before writing test cases, ALWAYS:
1. Call get_test_type_guidelines with the requested test type and follow those guidelines.
2. Call generate_test_case_ids with the requested number of test cases and use exactly those IDs.
3. Call get_priority_guidelines and use those rules to assign each Priority.

Rules:
- Write exactly one test case per ID you received, no more and no less.
- Each test case must be independent and repeatable.
- Do not duplicate scenarios.
- Use realistic, specific test data (e.g. "user@example.com", not "valid email").
- Expected results must be specific and verifiable.
- If the requirement is unclear, state your assumption in one line before the test cases.
- If the input is not a software requirement, politely reply that you can only
  generate test cases for software requirements.

Output format (Markdown, repeat for each test case, nothing else before or after
except an optional one-line assumption):

### TC_001
**Scenario:** <one-line description of what is being tested>

**Test Steps:**
1. <step>
2. <step>

**Test Data:** <specific data used>

**Expected Result:** <specific, verifiable outcome>

**Priority:** <High / Medium / Low>

---
"""


# Step 4: Create the agent.
# Streamlit reruns this whole script every time the user interacts with the page.
# @st.cache_resource makes sure the agent is created only once and then reused.
@st.cache_resource
def get_agent():
    return create_agent(
        model=MODEL,
        tools=[get_test_type_guidelines, generate_test_case_ids, get_priority_guidelines],
        system_prompt=SYSTEM_PROMPT,
    )


def build_user_prompt(requirement: str, test_type: str, count: int) -> str:
    """Step 5: Build the user prompt with a normal Python f-string (same as Part 1)."""
    return f"""
Generate {count} {test_type} test case(s) for the following requirement.

Requirement:
{requirement}

Test Type: {test_type}
Number of Test Cases: {count}
"""


def generate_test_cases(user_prompt: str) -> tuple[str, list[str]]:
    """Step 6: Send the prompt to the agent.
    Returns the final reply and a list of the tool calls the agent made."""
    result = get_agent().invoke({"messages": [{"role": "user", "content": user_prompt}]})

    tools_used = []
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            tools_used.append(f"{call['name']}({call['args']})")

    return result["messages"][-1].content, tools_used


# ---------------------------------------------------------------------------
# Step 7: Build the web page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="QA Test Case Generator", page_icon="🧪")

st.title("QA Test Case Generator")
st.write("Describe a feature, choose a test type, and get ready-to-use test cases.")

# Text area -> Requirement
requirement = st.text_area(
    "Requirement",
    placeholder="e.g. Users can log in with email and password. "
                "The password must be 8-20 characters. The account locks after 5 failed attempts.",
    height=150,
)

# Dropdown and number input side by side
col1, col2 = st.columns(2)
with col1:
    # Dropdown -> Test Type
    test_type = st.selectbox("Test type", TEST_TYPES)
with col2:
    # Number input -> Number of Test Cases
    count = st.number_input("Number of test cases", min_value=1, max_value=30, value=5, step=1)

# Button -> Generate Test Cases
if st.button("Generate test cases", type="primary"):
    if not requirement.strip():
        st.warning("Enter a requirement to generate test cases.")
    else:
        user_prompt = build_user_prompt(requirement.strip(), test_type, int(count))
        with st.spinner("Generating test cases..."):
            try:
                test_cases, tools_used = generate_test_cases(user_prompt)
                # Save the results in session_state so they stay on screen
                # when Streamlit reruns (for example, after clicking Download).
                st.session_state["test_cases"] = test_cases
                st.session_state["tools_used"] = tools_used
            except Exception as error:
                st.error(f"Could not generate test cases: {error}")

# Display the generated test cases on the screen
if "test_cases" in st.session_state:
    st.divider()
    st.subheader("Generated test cases")

    with st.expander("Tools used by the agent"):
        for call in st.session_state["tools_used"]:
            st.code(call, language=None)

    st.markdown(st.session_state["test_cases"])

    st.download_button(
        "Download test cases",
        data=st.session_state["test_cases"],
        file_name="test_cases.md",
        mime="text/",
    )