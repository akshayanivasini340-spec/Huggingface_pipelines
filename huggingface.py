import pandas as pd
from pathlib import Path
from transformers import pipeline

# ==========================================
# HUGGING FACE NLP PIPELINES
# ==========================================

print("=" * 60)
print("       HUGGING FACE NLP PIPELINES")
print("=" * 60)

# Create Dataset folder
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
DATASET_DIR.mkdir(exist_ok=True)

# ==========================================
# 1. CREATE INPUT CSV
# ==========================================

input_file = DATASET_DIR / "input_text.csv"

if not input_file.exists():
    input_data = {
        "Text": [
            "Artificial Intelligence is transforming the world. It is used in healthcare, education and business. Machine learning helps computers learn from data.",
            "Hugging Face provides pretrained models and tools for Natural Language Processing. It supports tasks such as translation, summarization and question answering."
        ],
        "Context": [
            "Artificial Intelligence is transforming the world. It is used in healthcare, education and business. Machine learning helps computers learn from data.",
            "Hugging Face is a company that provides pretrained models and tools for Natural Language Processing. It supports translation, summarization and question answering."
        ],
        "Question": [
            "Where is Artificial Intelligence used?",
            "What does Hugging Face provide?"
        ]
    }

    pd.DataFrame(input_data).to_csv(
        input_file, index=False, encoding="utf-8"
    )

df = pd.read_csv(input_file)

print("\nInput dataset loaded successfully!")
print(df)


# ==========================================
# 2. TEXT SUMMARIZATION
# ==========================================

print("\n1. TEXT SUMMARIZATION")
print("-" * 60)

summarizer = pipeline(
    "text2text-generation",
    model="t5-small"
)

summary_results = []

for text in df["Text"]:
    result = summarizer(
        "summarize: " + str(text),
        max_new_tokens=50
    )
    summary_results.append(result[0]["generated_text"])

summary_df = pd.DataFrame({
    "Original Text": df["Text"],
    "Summary": summary_results
})

summary_df.to_csv(
    DATASET_DIR / "summarized_text.csv",
    index=False,
    encoding="utf-8"
)

print(summary_df)


# ==========================================
# 3. TRANSLATION
# ==========================================

print("\n2. TEXT TRANSLATION")
print("-" * 60)

french_translator = pipeline(
    "text2text-generation",
    model="Helsinki-NLP/opus-mt-en-fr"
)

spanish_translator = pipeline(
    "text2text-generation",
    model="Helsinki-NLP/opus-mt-en-es"
)

french_results = []
spanish_results = []

for text in df["Text"]:
    french = french_translator(str(text))
    spanish = spanish_translator(str(text))

    french_results.append(
        french[0]["generated_text"]
    )

    spanish_results.append(
        spanish[0]["generated_text"]
    )

translation_df = pd.DataFrame({
    "Original Text": df["Text"],
    "French Translation": french_results,
    "Spanish Translation": spanish_results
})

translation_df.to_csv(
    DATASET_DIR / "translated_text.csv",
    index=False,
    encoding="utf-8"
)

print(translation_df)


# ==========================================
# 4. SENTIMENT ANALYSIS
# ==========================================

print("\n3. SENTIMENT ANALYSIS")
print("-" * 60)

sentiment_analyzer = pipeline(
    "sentiment-analysis"
)

sentiment_results = []

for text in df["Text"]:
    result = sentiment_analyzer(str(text))

    sentiment_results.append({
        "Sentiment": result[0]["label"],
        "Confidence": result[0]["score"]
    })

sentiment_df = pd.DataFrame(sentiment_results)

sentiment_df.insert(
    0, "Original Text", df["Text"]
)

sentiment_df.to_csv(
    DATASET_DIR / "sentiment_results.csv",
    index=False,
    encoding="utf-8"
)

print(sentiment_df)


# ==========================================
# 5. TEXT CLASSIFICATION
# ==========================================

print("\n4. TEXT CLASSIFICATION")
print("-" * 60)

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

categories = [
    "Technology",
    "Education",
    "Business",
    "Healthcare"
]

classification_results = []

for text in df["Text"]:
    result = classifier(
        str(text),
        candidate_labels=categories
    )

    classification_results.append(
        result["labels"][0]
    )

print("Classification Results:")
for category in classification_results:
    print(category)


# ==========================================
# 6. QUESTION ANSWERING
# ==========================================

print("\n5. QUESTION ANSWERING")
print("-" * 60)

qa_pipeline = pipeline(
    "question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad"
)

qa_results = []

for index, row in df.iterrows():
    result = qa_pipeline(
        question=str(row["Question"]),
        context=str(row["Context"])
    )

    qa_results.append({
        "Question": row["Question"],
        "Answer": result["answer"],
        "Confidence": result["score"]
    })

qa_df = pd.DataFrame(qa_results)

qa_df.to_csv(
    DATASET_DIR / "qa_results.csv",
    index=False,
    encoding="utf-8"
)

print(qa_df)


# ==========================================
# 7. FINAL COMPARISON
# ==========================================

print("\nFINAL COMPARISON")
print("=" * 60)

comparison_df = pd.DataFrame({
    "Original Text": df["Text"],
    "Summary": summary_results,
    "French Translation": french_results,
    "Spanish Translation": spanish_results,
    "Sentiment": sentiment_df["Sentiment"],
    "Classification": classification_results,
    "Answer": qa_df["Answer"]
})

comparison_df.to_csv(
    DATASET_DIR / "final_comparison.csv",
    index=False,
    encoding="utf-8"
)

print(comparison_df)

print("\n" + "=" * 60)
print("ALL NLP TASKS COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nAll CSV files saved inside Dataset folder.")