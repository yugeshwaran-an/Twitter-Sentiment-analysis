import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading Twitter dataset...")
df = pd.read_csv("Twitter_Data.csv")
print("Dataset loaded successfully!")
print("\nFirst 10 rows:")
print(df.head(10))
print("\nDataset Shape:", df.shape)
print("\nColumn Information:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())

text_column = "clean_text"
sentiment_column = "category"

df = df.dropna(subset=[text_column, sentiment_column]).copy()
df = df[df[text_column].astype(str).str.strip() != ""].copy()
df = df.drop_duplicates()

sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}
df[sentiment_column] = pd.to_numeric(df[sentiment_column], errors="coerce")
df = df[df[sentiment_column].isin(sentiment_labels)].copy()
df[sentiment_column] = df[sentiment_column].map(sentiment_labels)

print("\nDataset Shape After Cleaning:", df.shape)
print("\nSentiment Counts:")
print(df[sentiment_column].value_counts())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

df["cleaned_text"] = df[text_column].apply(clean_text)
df = df[df["cleaned_text"] != ""].copy()

print("\nOriginal and Cleaned Text:")
print(df[[text_column, "cleaned_text"]].head())

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x=sentiment_column)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

df[sentiment_column].value_counts().plot(
    kind="pie", autopct="%1.1f%%", figsize=(6, 6))
plt.title("Sentiment Percentage")
plt.ylabel("")
plt.tight_layout()
plt.show()

df["tweet_length"] = df[text_column].astype(str).str.len()
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x=sentiment_column, y="tweet_length")
plt.title("Tweet Length by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Tweet Length")
plt.tight_layout()
plt.show()

print("\nTweet Length Summary:")
print(df["tweet_length"].describe())

df.to_csv("cleaned_twitter_training.csv", index=False)
print("\nCleaned dataset saved successfully!")

print("\n--- PRACTICAL ACTION RECOMMENDATIONS ---")

counts = df[sentiment_column].value_counts()
negative = counts.get("Negative", 0)
neutral = counts.get("Neutral", 0)
positive = counts.get("Positive", 0)

if negative >= neutral and negative >= positive:
    print("1. Improve customer support and resolve complaints quickly.")
    print("2. Identify common problems and improve products or services.")
    print("3. Collect customer feedback and monitor complaints regularly.")
elif positive >= negative and positive >= neutral:
    print("1. Maintain product and service quality.")
    print("2. Continue practices that customers appreciate.")
    print("3. Strengthen customer relationships and encourage feedback.")
else:
    print("1. Understand why customers have neutral opinions.")
    print("2. Improve products and services using customer feedback.")
    print("3. Communicate product benefits more clearly.")

print("\nAction recommendations generated successfully!")
print("\nTWITTER SENTIMENT ANALYSIS COMPLETED!")