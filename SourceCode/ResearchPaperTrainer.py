import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, EarlyStoppingCallback, IntervalStrategy

class ResearchPaperTrainer:

    def __init__(self):
        self.dataset = None
        self.split_dataset = None
        self.trainer = None
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.trainVal = None

    def LoadDataframe(self):
        data_path = "../data/arXiv_SecondStage_Final.csv"
        self.dataset = pd.read_csv(data_path)
        self.dataset = self.dataset.rename(columns={"reduced_articles": "input_text", "abstract": "target_text"})
        self.dataset['combined_text'] = "Summarize: "+self.dataset['input_text'] + " <|sep|> " + self.dataset['target_text']

        self.dataset = Dataset.from_pandas(self.dataset)

        self.split_dataset = self.dataset.train_test_split(test_size=0.1)

    def SetupModelandTokenizer(self):
        
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': ['<|sep|>'],
            'pad_token': '[PAD]'
        })
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def tokenize_function(self, dataPoint):
        tokenized = self.tokenizer(
            dataPoint['combined_text'],
            max_length=1024,
            truncation=True,
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def GenerateTrainVal(self):
        tokenized_datasets = self.split_dataset.map(self.tokenize_function, batched=True)
        self.trainVal = tokenized_datasets["train"], tokenized_datasets["test"]
    
    def Train(self):
        training_args = TrainingArguments(
            output_dir="/content/drive/MyDrive/ATS/results",
            evaluation_strategy= IntervalStrategy.STEPS,  # Evaluate at the end of each epoch
            eval_steps = 50,
            learning_rate=5e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=10,
            fp16=True,
            # logging_dir="./logs",
            report_to="none",
            metric_for_best_model = 'eval_loss',
            load_best_model_at_end=True,
            save_safetensors=False
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.trainVal[0],
            eval_dataset=self.trainVal[0],
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        self.trainer.train()

    def SaveModel(self):
        self.trainer.save_model("../model/ResearchPaperModel")
        self.tokenizer.save_pretrained("../model/ResearchPaperModel")


    
    def Setup(self):
        self.LoadDataframe()
        self.SetupModelandTokenizer()
        self.GenerateTrainVal()

        self.Train()
