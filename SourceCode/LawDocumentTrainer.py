import pandas as pd
# import numpy as np
# import nltk
# import pathlib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
# import warnings
from textblob import TextBlob # type: ignore
import joblib
import re

class LawDocumentTrainer:

    def __init__(self):
        # self.legal_dF = pd.read_csv('../data/legal_text_classification.csv')
        self.justice_df = pd.read_csv('../data/justice.csv')
        self.model_filename = '../model/LawDocumentModel/legal_model.pkl'
        self.model_pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
        self.trainTestSplit = None


    def DataPreprocessing(self):

        self.first_party_winner_column = 'first_party_winner'
        self.justice_df['facts'] = self.justice_df['facts'].fillna('')

        if self.first_party_winner_column in self.justice_df.columns: 
            self.justice_df[self.first_party_winner_column] = self.justice_df[self.first_party_winner_column].fillna(0)
            self.justice_df[self.first_party_winner_column] = self.justice_df[self.first_party_winner_column].astype(int)
        else:
            raise KeyError(f"Column '{self.first_party_winner_column}' does not exist in justice_df.")
        
    def get_sentiment(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    def classify_sentiment(self, polarity):
        if polarity > 0:
            return "Positive"
        elif polarity < 0:
            return "Negative"
        else:
            return "Neutral"
        
    def GenerateSentiment(self):
        self.justice_df['facts_sentiment'] = self.justice_df['facts'].apply(self.get_sentiment)
        self.justice_df['sentiment_label'] = self.justice_df['facts_sentiment'].apply(self.classify_sentiment)
        print(self.justice_df[['facts', 'facts_sentiment', 'sentiment_label']].head())

    
    def Train(self):
        X = self.justice_df['facts']
        y = self.justice_df[self.first_party_winner_column]

        self.trainTestSplit = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model_pipeline.fit(self.trainTestSplit[0], self.trainTestSplit[2])

    def GenerateClassificationReport(self):
        y_pred = self.model_pipeline.predict(self.trainTestSplit[1])
        print(classification_report(self.trainTestSplit[3], y_pred))

    def SaveModel(self):
        joblib.dump(self.model_pipeline, self.model_filename)

        print(f"Model saved to {self.model_filename}")

    def predict_judgment(self, facts, loaded_model): 
        prediction = loaded_model.predict([facts])
        return prediction[0]
    
    def extract_parties(self, facts):
        match = re.search(r"([\w\s,]+)\s+v(?:\.|s\.?)\s+([\w\s,]+)", facts, re.IGNORECASE)
        if match:
            first_party = match.group(1).strip()
            second_party = match.group(2).strip()
            return first_party, second_party

        match = re.search(r"In the case of ([\w\s]+) and ([\w\s]+)", facts, re.IGNORECASE)
        if match:
            first_party = match.group(1).strip()
            second_party = match.group(2).strip()
            return first_party, second_party

        match = re.search(r"([\w\s]+)\s+against\s+([\w\s]+)", facts, re.IGNORECASE)
        if match:
            first_party = match.group(1).strip()
            second_party = match.group(2).strip()
            return first_party, second_party

        return None, None
    
    def PredictJudgement(self, new_case_facts):
        loaded_model = joblib.load(self.model_filename) #load saved trained model
        predicted_judgment = self.predict_judgment(new_case_facts, loaded_model)
        first_party, second_party = self.extract_parties(new_case_facts)
        label_mapping = {0: "Second Party Wins", 1: "First Party Wins"}
        predicted_outcome = label_mapping[predicted_judgment]

        if first_party and second_party:
            print(f"Parties involved: \nFirst Party: '{first_party}' \nSecond Party: '{second_party}'")
        else:
            print("Could not extract party names from the facts.")
        print(f"The predicted judgment is: {predicted_outcome}")

    def Setup(self):
        self.DataPreprocessing()
        self.GenerateSentiment()
        self.Train()

        self.GenerateClassificationReport()
        self.SaveModel()
        new_case_facts = "After observing and interviewing a number of people synthesizing and using drugs in a two-county area in Kentucky, Branzburg, a reporter, wrote a story which appeared in a Louisville newspaper. On two occasions he was called to testify before state grand juries which were investigating drug crimes. Branzburg refused to testify and potentially disclose the identities of his confidential sources. Similarly, in the companion cases of In re Pappas and United States v. Caldwell, two different reporters, each covering activity within the Black Panther organization, were called to testify before grand juries and reveal trusted information. Like Branzburg, both Pappas and Caldwell refused to appear before their respective grand juries."


        self.PredictJudgement(new_case_facts)


if __name__ == '__main__':

    ldt = LawDocumentTrainer()
    ldt.Setup()





        
