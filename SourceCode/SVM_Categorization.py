from SVMClassifier import SVMClassifier

test_text = '''It was March 2018, and then-President Donald Trump was meeting with his Department of Veterans Affairs Secretary, Dr. David Shulkin, about how to reform veteran health care. But it was Hegseth, then a Fox News personality, whose opinion Trump really wanted.

Hegseth, now Trump’s nominee to serve as secretary of defense, had been a vocal and persistent advocate for veterans having unfettered access to private health care, rather than having to go through the VA to keep their benefits. He’s also lobbied for policies that would restrict VA care and believes veterans should ask for fewer government benefits.
“We want to have full choice where veterans can go wherever they want for care,” Hegseth told Trump on speakerphone as Shulkin listened, according to Shulkin’s 2019 memoir.

Trump’s pick to serve as the next VA secretary, Doug Collins, has also expressed support for greater privatization of veteran health care, which advocates characterize as giving veterans greater choice over their doctors. If veterans “want to go back to their own doctors, then so be it,” he told Fox News last month.

For Shulkin, a rare “holdover” from President Barack Obama’s administration to Trump’s, this was “the worst-case scenario” for veteran health care, and one he had repeatedly warned Hegseth against.
“Your version of choice would cost billions more per year, bankrupting the system,” Shulkin recalls telling Hegseth in his memoir. “How can we responsibly pursue this? Unfortunately, he didn’t want to engage at the level of budget and other aspects of day-to-day reality. He seemed to prefer his sound bites on television.”

If confirmed, Hegseth and Collins will have the opportunity to push for a dramatic overhaul of the military and veteran health care system, one that could significantly cut government health benefits for service members and veterans – many of which Hegseth says veterans should not be asking for at all.
'''

if __name__ == "__main__":

    ## Setting up Classifier
    svm = SVMClassifier()
    svm.Setup()
    svm.GenerateClassificationReport()

    ## Testing with totally new unseen data
    svm.PredictCategory(test_text)




