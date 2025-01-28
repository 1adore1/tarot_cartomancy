# Tarot Cartomancy



### Overview

A simple tarot card reading web application built with Streamlit, using Hugging Face Inference API for AI-powered interpretations. Try it: [tarotfate.streamlit.app](https://tarotfate.streamlit.app/)

## Features

- Draws three random tarot cards for a given question.
- Displays card images with corresponding names.
- Provides AI-generated structured interpretations based on the drawn cards.
- User-friendly interface powered by Streamlit.

## Installation

### Prerequisites

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/index)
- OpenCV, NumPy, Pandas

### Setup

Clone the repository:

```
git clone https://github.com/1adore1/tarot_cartomancy.git
cd tarot_cartomancy
```

Install dependencies:

```
pip install -r requirements.txt
```

## Running the App

Set up your HuggingFace API key in a `secrets.toml` file under `.streamlit/`:

```
[secrets]
API_KEY = "your_huggingface_api_key"
```

Then, start the application:

```
streamlit run src/main.py
```

## How It Works

1. The user enters a question.
2. The app selects three random tarot cards.
3. The cards' images and names are displayed.
4. The model generates an interpretation of the drawn cards based on the question.

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **ML Models**: `Qwen2.5-Coder-32B-Instruct`, `bart-large-cnn`
