import pandas as pd
from transformers import pipeline
from huggingface_hub import login
import os
import torch
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


MY_HF_TOKEN = ""

INPUT_MOVIES_CSV = "TMDB_top_rated_movies_with_financials_by_id.csv"
OUTPUT_MOVIES_SENTIMENT_CSV = "TMDB_movies_with_overview_sentiment.csv"



# Dictionaries for combining GoEmotions emotions into 8 basic categories of a Cheat
# If a certain class is not specified, its probability will be considered 0 for the base emotion
PLUTCHIK_EMOTION_MAPPING = {
    "Joy": ['joy', 'amusement', 'excitement', 'optimism', 'pride', 'relief'],
    "Sadness": ['sadness', 'grief', 'disappointment', 'remorse'],
    "Anticipation": ['desire', 'curiosity'],
    "Surprise": ['surprise', 'realization', 'confusion'],
    "Fear": ['fear', 'nervousness', 'embarrassment'],
    "Anger": ['anger', 'annoyance', 'disapproval'],
    "Trust": ['caring', 'love', 'gratitude', 'admiration', 'approval'],
    "Disgust": ['disgust']
}


try:
    login(token=MY_HF_TOKEN)
except Exception as e:
    print(e)

model_name = "SamLowe/roberta-base-go_emotions"
emotion_classifier = None

if torch.cuda.is_available():
    emotion_classifier = pipeline("text-classification", model=model_name, top_k=None, device=0)
else:
    emotion_classifier = pipeline("text-classification", model=model_name, top_k=None, device='cpu')



def analyze_overview_sentiment(overview_text: str, classifier) -> dict:
    """
    Analyzes the text of the description of the film and calculates 8 values of basic emotions according to Plutchick.

        Args:
            overview_text (str): The text of the movie description.
            classifier: The uploaded emotionclassifier model.

        Returns:
            dict: A dictionary with 8 basic emotions and their average probabilities.
                  Returns a dictionary with zeros for all emotions in case of an error or an empty text.
        """
    # Initialize the dictionary for the result with zeros
    result_emotions = {plutchik_emotion: 0.0 for plutchik_emotion in PLUTCHIK_EMOTION_MAPPING.keys()}

    if not overview_text or pd.isna(overview_text) or str(overview_text).strip() == "":
        return result_emotions

    try:
        predictions = classifier(overview_text)

        # predictions[0] is a list of dictionaries, each of which {'label': 'emotion', 'score': 0.X}
        emotion_scores = {pred['label']: pred['score'] for pred in predictions[0]}

        # Calculate the average probability for each basic emotion of the Cheat
        for plutchik_emotion, go_emotions_list in PLUTCHIK_EMOTION_MAPPING.items():
            sum_scores = 0.0
            count_scores = 0
            for go_emotion in go_emotions_list:
                if go_emotion in emotion_scores:
                    sum_scores += emotion_scores[go_emotion]
                    count_scores += 1

            if count_scores > 0:
                result_emotions[plutchik_emotion] = sum_scores / count_scores
            else:
                result_emotions[plutchik_emotion] = 0.0

    except Exception as e:
        print(f"  Error in '{overview_text[:100]}...': {e}")
        return result_emotions

    return result_emotions


if __name__ == "__main__":
    if emotion_classifier is None:
        exit()


    try:
        df = pd.read_csv(INPUT_MOVIES_CSV)
        print(f"Downloaded {len(df)} films out of '{INPUT_MOVIES_CSV}'")
        print("Columns presented", df.columns.tolist())


        # We create new columns for each basic emotion of the Plutchick, initializing them with NaN
        for emotion_name in PLUTCHIK_EMOTION_MAPPING.keys():
            if emotion_name not in df.columns:
                df[emotion_name] = np.nan

        processed_count = 0

        for index, row in df.iterrows():
            overview_text = row['overview']
            movie_title = row.get('title', f"Фильм ID: {row.get('id', 'N/A')}")

            # Check if the emotion data is already there (i.e. not NaN)
            # This allows you to continue if the script was interrupted
            first_plutchik_emotion_col = list(PLUTCHIK_EMOTION_MAPPING.keys())[0]
            if pd.notna(row[first_plutchik_emotion_col]):
                processed_count += 1
                continue

            print(f"Analysis of {processed_count + 1}/{len(df)}: '{movie_title}'")

            emotion_results = analyze_overview_sentiment(overview_text, emotion_classifier)

            # # Writing the results to the Data Frame
            for emotion_name, score in emotion_results.items():
                df.loc[index, emotion_name] = score

            processed_count += 1

        print(df[['title', 'overview'] + list(PLUTCHIK_EMOTION_MAPPING.keys())].head())

        df.to_csv(OUTPUT_MOVIES_SENTIMENT_CSV, index=False, encoding='utf-8')
        print(f"\nFeatures are stored in'{OUTPUT_MOVIES_SENTIMENT_CSV}'")

    except Exception as e:
        print(f"Therer is an error: {e}")
