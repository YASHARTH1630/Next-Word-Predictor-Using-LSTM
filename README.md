# Next Word Prediction using LSTM

A deep learning based Next Word Prediction model built using TensorFlow and LSTM networks. The model is trained on Shakespeare's *Hamlet* text corpus and predicts the most probable next word for a given input sequence.

---

## Features

* Text preprocessing using Keras Tokenizer
* Sequence generation for language modeling
* LSTM based neural network architecture
* Word prediction using softmax probabilities
* Streamlit integration for deployment
* Trained on Shakespeare text data

---

## Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Streamlit

---

## Project Structure

```text
├── app.py
├── next_word.keras
├── tokenizer.pickle
├── hamlet.txt
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## Dataset

The model is trained using the text corpus from:

```text
hamlet.txt
```

Text preprocessing includes:e

* Lowercasing
* Tokenization
* Sequence creation
* Padding sequences

---

## Model Architecture

```python


model = Sequential()

model.add(Input(shape=(max_sequence_len - 1,)))

model.add(Embedding(
    input_dm=total_words,
    output_dim=10
))

model.add(LSTM(150))

model.add(Dropout(0.3))

model.add(Dense(
    total_words,
    activation='softmax'
))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)




## Running the Project

Clone the repository:

```bash
git clone github.com/YASHARTH1630/Next-Word-Predictor-Using-LSTM
cd next-word-predictor-using-lstm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit app:

```bash
streamlit run app.py
```

---

## Deployment

The application can be deployed using:

* Streamlit Community Cloud

Required files for deployment:

```text
app.py
requirements.txt
runtime.txt
next_word.keras
tokenizer.pickle
```

---

## Future Improvements

* Better text corpus
* GRU/Bidirectional LSTM implementation
* Transformer based language model
* Sentence generation
* Improved UI

---

## Author

Yasharth Yash
