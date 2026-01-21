# Data-Preprocessing-and-Embeddings

📊 Data Preprocessing and Text Embeddings (NLP)
🔍 Project Overview

This repository focuses on Natural Language Processing (NLP) fundamentals, covering the complete pipeline from raw text to machine-learning-ready numerical representations.

The project demonstrates:

- How raw text data is cleaned and normalized
- How text is converted into numerical embeddings
- How embeddings are used for text classification using ML models

This repository is designed for learning and experimentation, especially for beginners in Machine Learning, NLP, and Generative AI.

## 📁 Repository Structure

```text
Data-Preprocessing-and-Embeddings/
│
├── Text-Preprocessing.ipynb
│   └── Text cleaning, tokenization, stopword removal, lemmatization
│
├── Text-Representation_Word Embeddings-1.ipynb
│   └── Bag of Words (BoW) and TF-IDF vectorization
│
├── Text-Representation_Word Embeddings-2.ipynb
│   └── Dense word embeddings and semantic representations
│
├── Text_Classification_using_ML.ipynb
│   └── Sentiment classification using ML models
│
├── IMDB Dataset.csv
│   └── Movie reviews dataset for sentiment analysis
│
├── GOT SCRIPT.txt
│   └── Raw Game of Thrones script for NLP preprocessing
│
└── README.md


📌 Datasets Used
📄 IMDB Dataset (IMDB Dataset.csv)

-Contains movie reviews
-Used for:

 --Text preprocessing
 --Feature extraction
 --Sentiment classification
 
📄 Game of Thrones Script (GOT SCRIPT.txt)

-Raw textual script data
-Used to:
 --Apply preprocessing techniques
 --Generate word embeddings
 --Understand real-world noisy text


🛠️ Techniques Implemented
🔹 Text Preprocessing
 Implemented essential NLP cleaning techniques such as:

-Lowercasing text
-Removing punctuation & special characters
-Tokenization
-Stopword removal
-Stemming / Lemmatization
-Removing extra spaces & noise

📌 Goal: Convert raw, unstructured text into clean and meaningful tokens.

🔹 Text Representation (Word Embeddings)
I explored multiple methods to convert text into numbers:

📘 Notebook 1: Word Embeddings – Part 1
-Bag of Words (BoW)
-TF-IDF (Term Frequency – Inverse Document Frequency)
-Vocabulary creation
-Sparse vector representation
📌 Goal: Understand classical text vectorization methods.

📘 Notebook 2: Word Embeddings – Part 2
-Dense vector representations
-Word-level embeddings
-Understanding semantic similarity between words
📌 Goal: Learn why embeddings are better than simple word counts.

🔹 Text Classification using Machine Learning
In this notebook, you:
-Used preprocessed text features
-Applied ML algorithms for classification
-Trained models on sentiment-based text data
-Evaluated model performance
📌 Goal: Use text embeddings in real ML pipelines.

🧰 Libraries & Tools Used
The project uses standard NLP and ML libraries:
pandas – data handling
numpy – numerical operations
nltk – text preprocessing
scikit-learn – ML models & vectorization
matplotlib / seaborn – visualization
re – text cleaning with regex 


🚀 How to Run This Project
1️⃣ Clone the repositor
url - git clone https://github.com/Ujjval009/Data-Preprocessing-and-Embeddings.git
2️⃣ Open the project in VS Code or Jupyter
cd Data-Preprocessing-and-Embeddings
jupyter notebook

👨‍💻 Author
Ujjval Sharma
Engineering Student | NLP & ML Learner
GitHub: https://github.com/Ujjval009