from datasets import load_dataset
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration

class PreprocessCNN:

    def __init__(self):
        self.dataset = None
        self.trainValTest = None
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        # self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    def IsDataValid(self, dataPoint):
        token = self.tokenizer(dataPoint['article'], return_tensors="pt")['input_ids']
        size = len(token.squeeze())
        return size > 900 and size < 1024

    def LoadDataset(self):
        dataset = load_dataset("cnn_dailymail", "3.0.0")
        train_data = dataset["train"].shuffle(seed=42)
        val_data = dataset['validation'].shuffle(seed=42)
        test_data = dataset['test'].shuffle(seed=42)

        self.trainValTest = train_data, val_data, test_data

    def IsSummaryLengthGreaterThan128(self, row):
        token = self.tokenizer(row['summary'], return_tensors="pt")['input_ids']
        return len(token.squeeze()) < 128

    def CreateDataFrame(self):
        df_dict = {"article":[],
           "summary":[]}
        print("Processing Train Dataset")
        i = 0
        length = len(self.trainValTest[0])
        for dataPoint in self.trainValTest[0]:
            if i%(length//20) == 0:
                print(f"Percentage completed: {(i*100)/(length)}%")
            if self.IsDataValid(dataPoint):
                df_dict['article'].append(dataPoint['article'])
                df_dict['summary'].append(dataPoint['highlights'])
            i+=1


        print("Processing Test Dataset")
        i = 0
        length = len(self.trainValTest[1])
        for dataPoint in self.trainValTest[1]:
            if i%(length//20) == 0:
                print(f"Percentage completed: {(i*100)/(length)}%")
            if self.IsDataValid(dataPoint):
                df_dict['article'].append(dataPoint['article'])
                df_dict['summary'].append(dataPoint['highlights'])
            i+=1

        i = 0
        print("Processing Test Dataset")
        length = len(self.trainValTest[2])
        for dataPoint in self.trainValTest[2]:
            if i%(length//20) == 0:
                print(f"Percentage completed: {(i*100)/(length)}%")
            if self.IsDataValid(dataPoint):
                df_dict['article'].append(dataPoint['article'])
                df_dict['summary'].append(dataPoint['highlights'])
            i+=1

        newsArticles_dF = pd.DataFrame(df_dict)
        mask = newsArticles_dF.apply(self.IsSummaryLengthGreaterThan128,axis=1)
        self.dataset = newsArticles_dF[mask]

    def Setup(self):
        self.LoadDataset()
        self.CreateDataFrame()

    def SaveDataFrame(self):
        self.dataset.to_csv('CNN_Reduced_128_Full.csv')

    
if __name__ == '__main__':

    preprocessCNN = PreprocessCNN()
    
    preprocessCNN.Setup()
    preprocessCNN.SaveDataFrame()