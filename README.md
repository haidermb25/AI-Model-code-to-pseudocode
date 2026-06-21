# AI-Model-code-to-pseudocode

A Transformer-based deep learning model that converts source code into pseudocode. The project is built using PyTorch and provides an interactive interface through Streamlit.

## Features

* Transformer-based Seq2Seq model
* Positional Encoding
* Streamlit web application
* Automatic pseudocode generation
* GPU support (CUDA)

## Technologies Used

* Python
* PyTorch
* Streamlit
* JSON

## Project Structure

```text id="u14f4s"
AI-Model-code-to-pseudocode
│
├── app.py
├── vocabulary.json
├── transformer_epoch_8.pth
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash id="k3ek93"
git clone https://github.com/haidermb25/AI-Model-code-to-pseudocode.git
```

Move into the project directory:

```bash id="38s9do"
cd AI-Model-code-to-pseudocode
```

Install dependencies:

```bash id="0vnl0r"
pip install -r requirements.txt
```

## Running the Application

```bash id="jw7l0f"
streamlit run app.py
```

## Example

Input code:

```cpp id="zvhzv8"
int a, b;
cin >> a >> b;
cout << a + b;
```

Generated pseudocode:

```text id="mbo55u"
READ a
READ b
PRINT a + b
```

## Future Improvements

* Support multiple programming languages
* Improve model accuracy
* Add syntax highlighting
* Allow export of generated pseudocode

## Author

Ali Haider

* Software Engineer
* AI and Machine Learning Enthusiast

GitHub: https://github.com/haidermb25
