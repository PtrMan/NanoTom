# script to convert text to a databse format using 'language models'

import torch

from transformers import BloomTokenizerFast, BloomForCausalLM, pipeline

#modelName="bigscience/bloom-560m"
#modelName="bigscience/bloom-3b"
modelName="bigscience/bloom-7b1"


torch_dtype=torch.bfloat16
#torch_dtype=torch.float32

print("load model...")
model_bloom = BloomForCausalLM.from_pretrained(modelName, low_cpu_mem_usage=True)
print("load tokenizer...")
tokenizer_bloom = BloomTokenizerFast.from_pretrained(modelName)
print(f"build pipeline (sel dtype={type(torch_dtype)})...")
generator_bloom = pipeline(task="text-generation", model=model_bloom, tokenizer=tokenizer_bloom,   device=0, torch_dtype=torch_dtype)
print("...done")








from PromptModuleA import *
from TextUtils import *


def readTextFileLines(path):
    with open(path, 'r') as f:
        return f.readlines()

def readTextFile(path):
    return "\n".join(readTextFileLines(path))





# /param fromWikipedia is the text from wikipedia and thus "clean"?
def convRawTextToSentences(rawText, fromWikipedia):
    if not fromWikipedia:
        # clean up by concatenating linefeeds of sentences
        rawText = joinSentences(rawText)

    import re

    # regex to remove souces from wikipeda html output
    p = re.compile(r'\[\d+\]')
    text0 = re.sub(p, " ", rawText)


    # code to split sentences
    replacedPatterns = ["cf.","etc.",   "U.S.",    "2.0", "1.0", "e.g.","v.s.","i.e.",'et al.']
    t = text0
    for idx in range(len(replacedPatterns)):
        iReplacedPattern=replacedPatterns[idx]
        t = t.replace(iReplacedPattern,"MAGICAAA"+str(idx),5000000)
    
    

    
    
    
    # code to handle abbreviations
    import re
    ppA = re.compile(r"[\ ](\S)\.")
    ppB = re.compile(r"#§§(\S)\.")
    #text5 = "Tom E. D. Evilil was GREAT. X.Y.D. ."
    text5 = t
    text6 = re.sub(ppA, r" \1#§§", text5)
    #text6 = text5

    # repeat until we get fixpoint
    text6last = text6
    while True:
        #print(text6) DBG
        text6 = re.sub(ppB, r"#§§\1#§§", text6)
        if text6 == text6last:
            break
        text6last=text6
    
    
    
    
    t = text6


    t = t.replace(".",".\n",500000)

    for idx in range(len(replacedPatterns)):
        iReplacedPattern=replacedPatterns[idx]
        t = t.replace("MAGICAAA"+str(idx),iReplacedPattern,5000000)
    
    t = t.replace("#§§",".",50000000)




    
    sentences1 = t.split("\n")
    return sentences1




# 

retrievedArticles2 = []





articlesAsTxt0 = """Goods
Product (business)
Consumer
Rust (programming language)
ASP.NET
Open-source software
Software
High-level programming language"""


# Neural network

retrievedArticles2.append(None)

for iv in articlesAsTxt0.split('\n'):
    retrievedArticles2.append(iv)



for retrievedWikiArticle in retrievedArticles2:

    print(f"parse wikipedia: {retrievedWikiArticle}")
    
    
    if retrievedWikiArticle is None:
        sentences0 = readTextFile('./webRetrieved/induction.txt')
    else:
        import wikipediaapi
        wiki_wiki = wikipediaapi.Wikipedia(
                language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI)

        p_wiki = wiki_wiki.page(retrievedWikiArticle)
        #print(p_wiki.text)
        #sentences0 = p_wiki.summary
        sentences0 = p_wiki.text

    



    sentences0 = convRawTextToSentences(sentences0, retrievedWikiArticle is not None)





    sentences0 = list(map(lambda iSen:iSen.strip(), sentences0)) # remove spaces 

    sentences0 = list(filter(lambda iSen:len(iSen)>6, sentences0)) # remove empty and nonsense lines



    if retrievedWikiArticle is None:
        fout0 = open(f"out_7tgt_0014__paper__PeiWang induction.txt", 'w') 
    else:
        fout0 = open(f"out_7tgt_0012__wikipedia  {retrievedWikiArticle}.txt", 'w') 


    # NLP module to simplify complicated sentences to simpler statements
    class SimplifierModule(object):
        def __init__(self):
            pass

        # /param sentences0 array with sentences to get processed
        def doWork(self, sentences0):
            verbosityLevel = 1

            # list of raw sentences at level0
            sentencesLevel0 = sentences0

            # level1 sentences, which are simpler
            sentencesLevel1ByLevel0Idx = {}

            sentenceLevel0Idx = 0
            
            print("")
            
            # process sentences at level0 and split them up into easier to understand sentences to level1
            cnt0 = 0
            for iSentence in sentences0:
                print(f"{cnt0}/{len(sentences0)}~{iSentence}",end="\r")
                
                cnt0=cnt0+1
                
                if verbosityLevel >= 2:
                    print("")
                    print(f"{iSentence}")

                ### commented because it's not very useful in cutting down the complexity of the sentences
                ###p = ProcessSplit0()
                ###resSentencesLevel1 = p.process(generator_bloom, iSentence)
                #sentencesLevel1ByLevel0Idx[sentenceLevel0Idx] = resSentencesLevel1
                ###print(f"ProcessSplit0={resSentencesLevel1}")





                p = ProcessTextToQa3()
                resSentencesLevel1 = p.process(generator_bloom, iSentence)

                #sentencesLevel1ByLevel0Idx[sentenceLevel0Idx] = resSentencesLevel1

                
                if verbosityLevel >= 2 and retrievedWikiArticle is not None:
                    print(f"   wikipedia: {retrievedWikiArticle}")
                
                if verbosityLevel >= 2:
                    print(f"ProcessTextToQa3={resSentencesLevel1}")

                fout0.write("\n\n")

                for iResLine in resSentencesLevel1:
                    # emit original text and result line
                    fout0.write(f"{iSentence} #%%^ {iResLine}\n")
                    fout0.flush()

                sentenceLevel0Idx = sentenceLevel0Idx+1


    simplifierModuleInst = SimplifierModule()
    simplifierModuleInst.doWork(sentences0)
