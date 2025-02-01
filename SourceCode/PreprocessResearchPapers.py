from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import pandas as pd
import torch
import random

class PreprocessResearchPapers:

    def __init__(self):
        self.model_name = "google/pegasus-large"
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name)
        self.arXivdF = pd.read_csv("../data/arXiv_Processed_10k.csv")

    def ReduceTextLength(self, text):
        # Tokenize Text
        with torch.inference_mode():
            # model.to("cuda")
            inputs = self.tokenizer(text, return_tensors="pt")

            i = 0
            chucks = []
            while i*1024 <= len(inputs["input_ids"][0]):
                chunk = inputs["input_ids"][0, i*1024:(i+1)*1024 ]
                chucks.append(torch.unsqueeze(chunk, dim=0))
                i = i+1

            summarized_text = ''
            sum_len = 0
            for chunk in chucks:
                # Generate a summary
                # chunk = chunk.to("cuda")
                summary_ids = self.model.generate(
                    chunk,
                    max_length=256,
                    min_length=200,  
                    length_penalty=1.0,
                    num_beams=4,
                    early_stopping=True
                )
                sum_len += len(summary_ids[0])

                # Decode the summary
                summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summarized_text += summary
            print(f"Summary Token length {sum_len}")
            return summarized_text
        
    def TextReduction2ndRound(self, text):
        # Tokenize Text
        with torch.inference_mode():
            # model.to("cuda")

            inputs = self.tokenizer(text, return_tensors="pt")

            if len(inputs["input_ids"][0]) > 2000:

                i = 0
                chucks = []
                while i < 2:
                    chunk = inputs["input_ids"][0, i*1024:(i+1)*1024 ]
                    chucks.append(torch.unsqueeze(chunk, dim=0))
                    i = i+1


                summarized_text = ''
                sum_len = 0
                for chunk in chucks:
                    # Generate a summary
                    chunk = chunk.to("cuda")
                    summary_ids = self.model.generate(
                        chunk,
                        max_length=512,  
                        min_length=350,  
                        length_penalty=1.0,
                        num_beams=4,
                        early_stopping=True
                    )
                    sum_len += len(summary_ids[0])

                    # Decode the summary
                    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    summarized_text += summary
                print(f"Summary Token length {sum_len}")
                return summarized_text

            elif len(inputs["input_ids"][0]) > 1000 and len(inputs["input_ids"][0]) < 2000:
                print("between 1000 nd 2000")
                random_number = random.randint(-100, 20)
                chunk = inputs["input_ids"][0, 0:(1000+random_number)]

                # Decode the summary
                summary = self.tokenizer.decode(chunk, skip_special_tokens=True)
                print(f"Summary Token length {len(chunk)}")
                return summary

            else:
                print("less than 1000")
                return self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    

    def SkipRow(self, text):
        input = self.tokenizer(text, return_tensors="pt")
        if len(input['input_ids'][0]) > 6140:
            return True
        return False

    def Preprocess(self):
        self.model.eval()
        arXiv_Reduced = {"reduced_articles": [],
                        "article_id": [],
                        "abstract":[]}

        for i in range(0,len(self.arXivdF["article"])):

            if self.SkipRow(self.arXivdF["article"][i]):
                print(f"Skipping row: {i}")
                continue

            arXiv_Reduced["reduced_articles"].append( self.ReduceTextLength(self.arXivdF["article"][i]))
            arXiv_Reduced["article_id"].append(i)
            arXiv_Reduced["abstract"].append(self.arXivdF["abstract"][i])

            print(f"{i} row processed")


            if len(arXiv_Reduced['reduced_articles']) % 50 == 0:
                reduced_df = pd.DataFrame(arXiv_Reduced)
                reduced_df.to_csv(f"/content/drive/MyDrive/ATS/data/arXiv_{i}_ReducedText.csv", index=False)


if __name__ == '__main__':
    preprocessResearchPapers = PreprocessResearchPapers()

    preprocessResearchPapers.Preprocess()