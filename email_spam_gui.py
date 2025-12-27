import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


data = pd.read_csv("emails.csv")
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

X = data['text']
y = data['label']


vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(X)


model = MultinomialNB()
model.fit(X_tfidf, y)


def check_email():
    email_text = text_entry.get("1.0", tk.END).strip()

    if email_text == "":
        messagebox.showwarning("Warning", "Please enter email text")
        return

    email_tfidf = vectorizer.transform([email_text])
    prediction = model.predict(email_tfidf)
    probability = model.predict_proba(email_tfidf)[0]

    spam_prob = probability[1] * 100
    ham_prob = probability[0] * 100

    
    if spam_prob > 60:
        result_label.config(
            text=f"SPAM EMAIL\nConfidence: {spam_prob:.2f}%",
            fg="red"
        )
    else:
        result_label.config(
            text=f"NOT SPAM (HAM)\nConfidence: {ham_prob:.2f}%",
            fg="green"
        )

def clear_text():
    text_entry.delete("1.0", tk.END)
    result_label.config(text="")


root = tk.Tk()
root.title("Email Spam Detector")
root.geometry("520x430")
root.config(bg="#f2f2f2")


title_label = tk.Label(
    root, text="Email Spam Detection System",
    font=("Arial", 16, "bold"), bg="#f2f2f2"
)
title_label.pack(pady=10)


text_entry = tk.Text(root, height=8, width=55)
text_entry.pack(pady=10)


button_frame = tk.Frame(root, bg="#f2f2f2")
button_frame.pack(pady=10)

check_button = tk.Button(
    button_frame, text="Check Email",
    font=("Arial", 11), 
    bg="blue", fg="white",
    command=check_email
)
check_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(
    button_frame, text="Clear",
    font=("Arial", 11),
    bg="gray", fg="white",
    command=clear_text
)
clear_button.grid(row=0, column=1, padx=10)


result_label = tk.Label(
    root, text="",
    font=("Arial", 14, "bold"),
    bg="#f2f2f2"
)
result_label.pack(pady=15)


root.mainloop()

