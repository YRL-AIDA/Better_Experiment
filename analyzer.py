PROMPTS = [
    # 1. Zero-shot Prompting
    {
        "name": "zero_shot",
        "system": [
            "You are a data processing assistant.",
            "Your task is to identify table headers and return their positions in JSON format."
        ],
        "user": """Analyze the following table and determine which cells are headers.
Return your answer as JSON in this exact format:
{{"headers": [{{"row": 0, "col": 0}}, {{"row": 0, "col": 1}}]}}

Table:
{table_text}"""
    },
    
    # 2. Few-shot Prompting
    {
        "name": "few_shot",
        "system": [
            "You are a table structure recognition expert.",
            "You identify header cells and return their coordinates in JSON format."
        ],
        "user": """Your task is to identify table header cells.

Example 1:
Table:
| ID | Name | Age |
| 1 | Ivan | 25 |
| 2 | Anna | 30 |
Headers: {{"headers": [{{"row": 0, "col": 0}}, {{"row": 0, "col": 1}}, {{"row": 0, "col": 2}}]}}

Example 2:
Table:
| Product | Price |
| Category | Electronics |
| Laptop | 1200 |
Headers: {{"headers": [{{"row": 0, "col": 0}}, {{"row": 0, "col": 1}}]}}

Now identify the headers for this table:
{table_text}

Return JSON format: {{"headers": [{{"row": X, "col": Y}}, ...]}}"""
    },
    
    # 3. Role Prompting
    {
        "name": "role_prompting",
        "system": [
            "You are a leading data processing engineer with 20 years of experience in ETL processes.",
            "You specialize in automatic recognition of complex table structures and noisy CSV files.",
            "You excel at identifying headers even in non-standard table formats."
        ],
        "user": """As an expert in data structure analysis, examine this table and identify all header cells.
Consider data types, semantic meaning, and structural patterns.

Table:
{table_text}

Return your analysis as JSON: {{"headers": [{{"row": X, "col": Y}}, ...]}}"""
    },
    
    # 4. Zero-Shot Chain-of-Thought
    {
        "name": "zero_shot_cot",
        "system": [
            "You are a systematic data analyst.",
            "You solve problems step-by-step using logical reasoning."
        ],
        "user": """Identify the headers in this table. Let's think step by step:
1. First, analyze the first row - what type of data does it contain?
2. Then, compare it with the second row - are there differences in data types or semantic meaning?
3. Check if any cells contain metadata or descriptive labels rather than data values.
4. Based on this analysis, determine which cells are headers.

Table:
{table_text}

Provide your step-by-step reasoning, then return the final answer as JSON: {{"headers": [{{"row": X, "col": Y}}, ...]}}"""
    },
    
    # 5. Self-Consistency
    {
        "name": "self_consistency",
        "system": [
            "You are a table structure analyst.",
            "You generate multiple hypotheses and select the most consistent answer."
        ],
        "user": """Generate 3 different interpretations of where the headers are in this table:
- Interpretation 1: Headers in the first row
- Interpretation 2: Headers in multiple rows
- Interpretation 3: Headers might be in a non-standard position

After analyzing all interpretations, choose the most logical and consistent option.

Table:
{table_text}

Provide your reasoning for each interpretation, then return the final answer as JSON: {{"headers": [{{"row": X, "col": Y}}, ...]}}"""
    },
    
    # 6. Least-to-Most Prompting
    {
        "name": "least_to_most",
        "system": [
            "You are a methodical data analyst.",
            "You solve complex problems by breaking them into simple subtasks."
        ],
        "user": """Solve this task step by step from simple to complex:

Step 1: List the data type of each column in the first 3 rows.
Step 2: Identify which rows contain descriptive labels vs actual data values.
Step 3: Determine if the first row differs in format or semantic meaning from subsequent rows.
Step 4: Based on steps 1-3, identify the final header cell positions.

Table:
{table_text}

Show your work for each step, then return final answer as JSON: {{"headers": [{{"row": X, "col": Y}}, ...]}}"""
    },
    
    # 7. Tree of Thoughts
    {
        "name": "tree_of_thoughts",
        "system": [
            "You are a multi-perspective table analyst.",
            "You evaluate different expert opinions to reach the best conclusion."
        ],
        "user": """Analyze this table from three expert perspectives:

Expert A: "Headers are in the first row only"
Expert B: "There are no explicit headers, they need to be inferred"
Expert C: "Headers span multiple rows (multi-level structure)"

For each expert:
1. Present their argument
2. Evaluate validity based on the actual table data
3. Identify strengths and weaknesses

Finally, make a verdict on which interpretation is correct.

Table:
{table_text}

Show the analysis for each expert, then return the final answer as JSON: {{"headers": [{{"row": X, "col": Y}}, ...]}}"""
    },
    
    # 8. ReAct (Reason + Act)
    {
        "name": "react",
        "system": [
            "You are an analytical agent that combines reasoning with verification actions.",
            "You use Thought-Action-Observation cycles to solve problems."
        ],
        "user": """Use the Thought-Action-Observation loop to identify headers:

Thought 1: What does the structure suggest about header location?
Action 1: Examine data types in each column of row 0 vs row 1
Observation 1: [Record findings]

Thought 2: Are there semantic clues indicating header cells?
Action 2: Check for descriptive text vs numeric/categorical data
Observation 2: [Record findings]

Thought 3: Based on observations, where are the headers?
Action 3: Compile final header positions

Table:
{table_text}

Show your complete reasoning loop, then return JSON: {{"headers": [{{"row": X, "col": Y}}, ...]}}"""
    },
    
    # 9. Self-Refine
    {
        "name": "self_refine",
        "system": [
            "You are a self-critical data analyst.",
            "You create drafts, critique them, and refine your answers iteratively."
        ],
        "user": """Work through this iterative refinement process:

DRAFT (Step 1): Make your initial identification of header cells.

CRITIQUE (Step 2): Review your draft answer:
- Did you include any regular data cells as headers by mistake?
- Did you miss any multi-row or multi-level headers?
- Are there edge cases or unusual patterns you overlooked?

REFINED ANSWER (Step 3): Based on your self-critique, provide an improved, corrected answer.

Table:
{table_text}

Show all three steps clearly, then return final JSON: {{"headers": [{{"row": X, "col": Y}}, ...]}}"""
    },
    
    # 10. Reflexion
    {
        "name": "reflexion",
        "system": [
            "You are an experienced table analyst with memory of past mistakes.",
            "You learn from previous errors to improve current performance."
        ],
        "user": """Important lessons from past errors:
- Don't mistake the first row of data for headers just because it contains text
- Check if numeric-looking cells might actually be header labels (e.g., "2023", "Q1")
- Multi-row headers are common - don't assume only one row can be headers
- Empty cells in the first row might indicate merged header cells

Given these lessons, carefully analyze this table and identify the actual headers:

Table:
{table_text}

Explain how you avoided past mistakes, then return JSON: {{"headers": [{{"row": X, "col": Y}}, ...]}}"""
    },
    
    # 11. OPRO (Optimization by Prompting)
    {
        "name": "opro",
        "system": [
            "You are a highly accurate table structure recognition system.",
            "Data integrity depends on your precise header identification."
        ],
        "user": """Take a deep breath and work on this task step by step.

Your goal is to identify table headers with maximum accuracy to ensure data integrity.
This is critical for the project's success.

Carefully examine each cell's content, position, and relationship to other cells.
Consider all possibilities before making your final determination.

Table:
{table_text}

Work methodically and return your answer as JSON: {{"headers": [{{"row": X, "col": Y}}, ...]}}"""
    },
    
    # 12. Chain-of-Table
    {
        "name": "chain_of_table",
        "system": [
            "You are a table operations specialist.",
            "You use sequential table operations to analyze structure."
        ],
        "user": """Perform these table operations sequentially:

Operation 1 - SELECT_ROWS(0, 1, 2): Extract first 3 rows
Operation 2 - ANALYZE_TYPES: Check data types in each column for these rows
Operation 3 - COMPARE_SEMANTIC: Compare semantic meaning of row 0 vs rows 1-2
Operation 4 - IDENTIFY_PATTERNS: Look for header-indicating patterns (capitalization, naming conventions, etc.)
Operation 5 - GET_HEADERS: Based on operations 1-4, extract header cell positions

Table:
{table_text}

Execute each operation and show results, then return JSON: {{"headers": [{{"row": X, "col": Y}}, ...]}}"""
    },
    
    # 13. Medprompt Style (Composite)
    {
        "name": "medprompt",
        "system": [
            "You are an ensemble table analysis system.",
            "You combine multiple reasoning strategies and select the best answer by consensus."
        ],
        "user": """Use a composite analysis approach:

1. RETRIEVE: Recall similar table structures from your training knowledge
2. REASON: Generate chain-of-thought analysis for this specific table
3. GENERATE: Create 3 different header identification hypotheses
4. ENSEMBLE: Compare all hypotheses and select the answer with highest confidence
5. VALIDATE: Verify the selected answer against table structure rules

Table:
{table_text}

Show your work for each phase, then return consensus answer as JSON: {{"headers": [{{"row": X, "col": Y}}, ...]}}"""
    }
]