# InboxGuard — Intelligent Email Spam Detection System

A machine learning web application that detects spam emails with 96.68% accuracy.

## Live Demo
Type any email content and instantly know if it's spam or not!

## How It Works
- Uses **Naive Bayes** algorithm with **TF-IDF** vectorization
- Trained on 5,000+ real emails
- Achieves **96.68% accuracy**

## Tech Stack
- **Python** — core programming language
- **Scikit-learn** — machine learning
- **Flask** — web framework
- **NLTK** — natural language processing
- **HTML/CSS** — frontend interface

## How to Run
1. Clone the repository
2. Install dependencies:

pip install -r requirements.txt
3. Train the model:

python src/train.py
4. Run the app:

python src/app.py
5. Open browser: http://127.0.0.1:5000

## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 96.68% |
| Algorithm | Naive Bayes |
| Dataset | SMS Spam Collection |

## Developer
Built by Nida as an independent portfolio project.
