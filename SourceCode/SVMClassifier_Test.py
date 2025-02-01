from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd

class SVMClassifier:

    def __init__(self):
        self.model = None
        self.dataset = None
        self.trainTestSplit = None
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.inputTrainTestVectors = None

    def GetDataset(self):
        news_articles = pd.read_csv("../data/CNN_Reduced_128_Full.csv")
        research_papers = pd.read_csv("../data/arXiv_Final.csv")
        law_documents = pd.read_csv("../data/LawDocuments.csv")

        research_papers.drop(['Unnamed: 0', 'abstract'], axis=1, inplace=True)
        news_articles.drop(['Unnamed: 0','summary'], axis=1, inplace=True)
        law_documents.drop('Unnamed: 0', axis=1, inplace=True)
        law_documents = law_documents.dropna()

        news_data = news_articles.sample(n=len(research_papers), random_state=42)
        research_data = research_papers.sample(n=len(research_papers), random_state=42)
        law_data = law_documents.sample(n=len(research_papers), random_state=42)

        news_data['label'] = 'News Article'
        research_data['label'] = 'Research Paper'
        law_data['label'] = 'Law Document'

        news_data.rename(columns={'article': 'text'}, inplace=True)
        research_data.rename(columns={'reduced_articles': 'text'}, inplace=True)
        law_data.rename(columns={'case_text': 'text'}, inplace=True)

        combined_data = pd.concat([news_data, research_data, law_data])
        self.dataset = combined_data.sample(n=len(combined_data), random_state=42)

    def EncodeLabels(self):
        self.dataset['label'] = self.dataset['label'].map({'News Article': 0, 'Research Paper': 1, 'Law Document': 2})

    def GetTrainTestSplit(self):
        self.trainTestSplit = train_test_split(self.dataset['text'], self.dataset['label'], test_size=0.2, random_state=42)

    def Vectorize_TFIDF(self):
        X_train_tfidf = self.vectorizer.fit_transform(self.trainTestSplit[0])
        X_test_tfidf = self.vectorizer.transform(self.trainTestSplit[1])
        self.inputTrainTestVectors = X_train_tfidf, X_test_tfidf

    def TrainSVM(self):
        svm_model = SVC(kernel='linear', random_state=42)
        print("Training SVM model")
        svm_model.fit(self.inputTrainTestVectors[0], self.trainTestSplit[2])
        self.model = svm_model

    def PredictCategory(self, input_text):
        input_tfidf = self.vectorizer.transform([input_text])
        prediction = self.model.predict(input_tfidf)[0]
        categories = {0: 'News Article', 1: 'Research Paper', 2: 'Law Document'}
        print(f"Prediction Result: {categories[prediction]}")

    def EvaluateModel(self):
        y_pred = self.model.predict(self.inputTrainTestVectors[1])
        print("Clasification Report:")
        print(classification_report(self.trainTestSplit[3], y_pred, target_names=['News Article', 'Research Paper', 'Law Document']))

    def Setup(self):
        self.GetDataset()
        self.EncodeLabels()
        self.GetTrainTestSplit()
        self.Vectorize_TFIDF()
        self.TrainSVM()

    def GenerateClassificationReport(self):
        self.EvaluateModel()