import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import torch
from rouge_score import rouge_scorer
import numpy as np

class SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    

class NewsArticlesTrainer:

    def __init__(self):
        self.dataset = None
        self.trainValTest = None
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

        self.trainValTestEncodings = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        self.trainLoader = None
        self.valLoader = None

    def GenerateTrainTestSplit(self):
        df = pd.read_csv('../data/CNN_Reduced_128_Full.csv')
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        self.trainValTest = train_df, val_df, test_df

    def tokenize_function(self, batch):
        inputs = self.tokenizer(
            batch['article'],
            max_length=1024,
            truncation=True,
            padding="max_length"
        )
        labels = self.tokenizer(
            batch['summary'],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        inputs['labels'] = labels['input_ids']
        return inputs

    def TokenizeData(self):
        train_data = self.trainValTest[0].to_dict(orient="list")
        val_data = self.trainValTest[1].to_dict(orient="list")
        test_data = self.trainValTest[2].to_dict(orient="list")

        train_encodings = self.tokenize_function(train_data)
        val_encodings = self.tokenize_function(val_data)
        test_encodings = self.tokenize_function(test_data)

        self.trainValEncodings = train_encodings, val_encodings, test_encodings

    def Train(self):

        train_dataset = SummarizationDataset(self.trainValEncodings[0])
        val_dataset = SummarizationDataset(self.trainValEncodings[1])

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)

        self.trainLoader = train_loader
        self.valLoader = val_loader

        optimizer = AdamW(self.model.parameters(), lr=5e-5)

        self.model.train()
        for epoch in range(5):
            loop = tqdm(train_loader, leave=True)
            for batch in loop:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loop.set_description(f"Epoch {epoch}")
                loop.set_postfix(loss=loss.item())
    
    def SaveModel(self):
        self.model.save_pretrained("../model/NewsArticleModel")
        self.tokenizer.save_pretrained("../model/NewsArticleModel")

    def EvaluateModel(self):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                val_loss += outputs.loss.item()

        print(f"Validation Loss: {val_loss / len(self.val_loader)}")


    def CalculateRougeScore(self):

        test_encodings = self.tokenize_function(self.trainValTest[2])
        test_dataset = SummarizationDataset(test_encodings)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        self.model.eval()

        generated_summaries = []
        ground_truths = self.trainValTest[2]['summary']  # Ground truth summaries

        with torch.no_grad():
            for batch in tqdm(test_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    min_length=50,
                    num_beams=4,  
                    length_penalty=2.0
                )

                generated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generated_summaries.extend(generated)

        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

        for gen, ref in zip(generated_summaries, ground_truths):
            scores = scorer.score(gen, ref)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        avg_rouge1 = np.mean(rouge1_scores)
        avg_rouge2 = np.mean(rouge2_scores)
        avg_rougeL = np.mean(rougeL_scores)

        print(f"ROUGE-1: {avg_rouge1:.4f}")
        print(f"ROUGE-2: {avg_rouge2:.4f}")
        print(f"ROUGE-L: {avg_rougeL:.4f}")

    def Setup(self):
        self.GenerateTrainTestSplit()
        self.TokenizeData()
        self.Train()
        self.EvaluateModel()
        self.CalculateRougeScore()
