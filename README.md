# 🤖 Natural Language KPI Chatbot with Azure OpenAI, SQL & LIDA

A smart Streamlit-based chatbot that converts executive-level natural language questions into SQL queries, fetches results from a live SQL Server database, and automatically generates insightful visualizations using [LIDA](https://github.com/VIDA-NYU/LIDA) and Azure OpenAI.

---

## 🚀 Features

- 🔍 **Ask in Plain English** – Converts business questions into SQL queries using LangChain + Azure OpenAI
- 📊 **Auto-Visualize Results** – Generates context-aware charts with LIDA based on query results
- 🧠 **Context-Aware AI** – Maintains conversation history for better understanding
- 🔌 **SQL Integration** – Connects directly to a Microsoft SQL Server using SQLAlchemy
- 📈 **Custom Table Views** – Lets users select specific columns to visualize and filter data
- 🗂️ **Clean UI** – Built using Streamlit for interactive, user-friendly interface

---

## 📸 Demo

> *Ask questions like:*
> - “What is the YTD for Investments KPI?”
> - “Show actual vs target for ROE KPI”
> - “Visualize revenue breakdown by quarter”


---

## 📦 Tech Stack

| Category        | Tech |
|-----------------|------|
| Backend LLM     | Azure OpenAI (ChatGPT) |
| Database Access | SQLAlchemy, pyodbc |
| Prompt Engine   | LangChain |
| Visualization   | LIDA (Language Interface for Data Analysis) |
| Frontend        | Streamlit |
| Helpers         | Pandas, PIL, Regex, Base64 |

---

## 🛠️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/shettyrohit0810/kpi-chatbot-azure-sql-lida.git
cd kpi-chatbot-azure-sql-lida
