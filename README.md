# Spam Email Detection 

##  Project Overview
This project detects whether an email is **Spam** or **Not Spam** using **Machine Learning and Natural Language Processing (NLP)** techniques.  
A simple **GUI application** is also provided for user interaction.

---

##  Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Tkinter

---

##  Dataset
- File: `emails.csv`
- Contains labeled email text as **spam** or **ham**
- Text data is preprocessed using NLP techniques

---

##  How It Works
1. Email text preprocessing (cleaning, tokenization)
2. Feature extraction using **TF-IDF**
3. Classification using **Machine Learning model**
4. Prediction displayed via GUI

---

##  Model Used
- **Multinomial Naive Bayes**
- Chosen for its efficiency with text classification problems

---

##  Results
- Achieved approximately **90% accuracy** on test data

---

## ▶ How to Run the Project
```bash

python email_spam_gui.py

