Sentiment Classification with DistilBERT
========================================

A fine-tuned DistilBERT model trained on Sentiment140 dataset for real-time sentiment analysis of tweets.

What This Is
------------

This is a complete machine learning project that:
1. Loads and cleans 1.6 million tweets from Sentiment140
2. Tokenizes text using BERT tokenizer
3. Fine-tunes DistilBERT for sentiment classification
4. Provides a web interface using Streamlit

The model achieves 87.12% accuracy on the test set.

Results
-------

Model: DistilBERT (Distilled BERT)
Dataset: Sentiment140 (1.58M tweets after cleaning)
Training Time: 11.5 hours on RTX 3050
Final Accuracy: 87.12%
F1 Score: 87.12%

The model learned across 3 epochs:
- Epoch 0: 86.51% accuracy
- Epoch 1: 87.12% accuracy (best)
- Epoch 2: 87.04% accuracy

Project Structure
-----------------

app.py         - Streamlit web app (main interface)
requirements.txt         - Python dependencies
.gitignore              - Git ignore rules
training.py    - Training of the model (not included in repo)
model.py       - Main model code

my_trained_bert_sentiment_best/  - Trained model weights
data/                            - Dataset (not included in repo)

How to Use
----------

Local Setup:

1. Install dependencies:
   pip install -r requirements.txt

2. Run the Streamlit app:
   streamlit run sentiment_app.py

3. The app opens at http://localhost:8501

4. Type any text and get instant sentiment prediction with confidence score

Testing:

Try these examples:
- Positive: "I absolutely love this! It's amazing and incredible!"
- Negative: "This is terrible and a complete waste of time."
- Mixed: "It was okay, nothing special but not bad either."

Training Details
----------------

Dataset Processing:
- Original size: 1.6 million tweets
- Removed 1,685 duplicate IDs
- Removed 18,534 duplicate texts (same text = no new learning)
- Final size: 1,581,466 examples
- Split: 80% train (1.265M), 20% test (316K)
- Labels: Balanced between negative (790K) and positive (791K)

Tokenization:
- Using bert-base-uncased tokenizer
- Max length: 128 tokens (standard for BERT)
- Padding: max_length (pads short texts)
- Truncation: True (cuts long texts)

Training:
- Model: DistilBERT (40% smaller, 60% faster than BERT)
- Optimizer: AdamW with learning rate 2e-5
- Warmup steps: 500
- Batch size: 32
- Loss function: Cross-entropy (automatic with model)
- Scheduler: Linear with warmup

Tech Stack
----------

PyTorch - Deep learning framework
Transformers - Model loading and training
Streamlit - Web interface
FastAPI - Optional REST API
Scikit-learn - Evaluation metrics
NumPy, Pandas - Data processing

Key Learnings
-------------
Tokenizer: Always explicitly move model to device before training
Format Issues: Use set_format('torch') for PyTorch DataLoaders
Label Names: BERT expects 'labels' not 'label' as column name
Colab Limitations: Manual training loop more reliable than Trainer class
DistilBERT: Good trade-off between speed and accuracy

Issues Faced and Solutions
---------------------------

Issue 1: GPU Not Being Used
- Symptom: 77 mins for training 10% data on supposed GPU
- Cause: Model loaded but not moved to device
- Fix: Added model.to(device) before training

Issue 2: Colab CUDA Errors
- Symptom: "CUDA kernel errors" during training
- Cause: Trainer class version conflicts in Colab
- Fix: Used manual training loop instead of Trainer

Issue 3: Torchvision Import Error
- Symptom: "Cannot import VideoReader from torchvision"
- Cause: Version mismatch between torch and torchvision
- Fix: Disabled torch format before DataLoader

Issue 4: NoneType Loss Error
- Symptom: "NoneType has no attribute backward"
- Cause: Dataset column named 'label' instead of 'labels'
- Fix: Renamed column to 'labels' (BERT expects this)

Future Improvements
-------------------

Possible enhancements:
- Test accuracy improvement
- Add batch prediction (upload CSV of tweets)
- Add confidence threshold filtering
- Create REST API with rate limiting
- Add support for other languages
- Implement model quantization for faster inference


Dependencies
------------

Core:
- torch>=2.0.0
- transformers>=4.30.0
- datasets>=2.0.0

Interface:
- streamlit>=1.20.0
- fastapi>=0.100.0
- uvicorn>=0.23.0

Data Processing:
- pandas>=2.0.0
- numpy>=1.24.0
- scikit-learn>=1.3.0

All included in requirements.txt

Contributing
------------

This is a portfolio project. Feel free to fork and experiment.

If you improve the model accuracy or find bugs, let me know.

Contact
-------

Built by TanX (Mechanical Engineering + ML)
IIT Kharagpur, 2026

License
-------

MIT License - use freely for learning and projects
