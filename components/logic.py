# app.py

import gradio as gr

# === Budget Calculator Functions ===

def calculate_savings(income: float, expenses: float) -> str:
    savings = income - expenses
    if savings < 0:
        return f"⚠️ Your expenses exceed your income by PKR {abs(savings):,.0f}!"
    return f"💰 You can save PKR {savings:,.0f} per month"

def estimate_annual_tax(annual_income: float) -> str:
    if annual_income <= 600_000:
        tax = 0
    elif annual_income <= 1_200_000:
        tax = (annual_income - 600_000) * 0.05
    elif annual_income <= 2_400_000:
        tax = 30_000 + (annual_income - 1_200_000) * 0.10
    elif annual_income <= 3_600_000:
        tax = 150_000 + (annual_income - 2_400_000) * 0.15
    else:
        tax = 330_000 + (annual_income - 3_600_000) * 0.20
    return f"🧾 Estimated Annual Tax: PKR {tax:,.0f}"

def calculate_budget_summary(income, expenses):
    savings_msg = calculate_savings(income, expenses)
    tax_msg = estimate_annual_tax(income * 12)  # Annual income
    return f"{savings_msg}\n{tax_msg}"

# === Gradio UI ===

with gr.Blocks() as demo:
    gr.Markdown("## 💸 Budget Calculator + Tax Estimator (Pakistan FY 2024-25)")

    with gr.Row():
        income = gr.Number(label="Monthly Income (PKR)", value=150000)
        expenses = gr.Number(label="Monthly Expenses (PKR)", value=125000)

    submit_btn = gr.Button("Calculate")
    output = gr.Textbox(label="Summary", lines=4)

    submit_btn.click(fn=calculate_budget_summary, inputs=[income, expenses], outputs=output)

# Run the app
demo.launch()