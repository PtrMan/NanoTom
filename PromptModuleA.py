# module with prompt processors

# prompt processor to 'understand' a sentence and split it into multiple simpler sentences
class ProcessSplit0(object):
    def __init__(self):
        pass
    
    def process(self, generator, argText):
        # built for: bloom-176b
        # works for bloom-7b1
        # ref https://gist.github.com/PtrMan/a4f0ffa734b6df6d0d16387f82e42303#file-0033-tool-explain-as-alien-works-txt-L1
        promptText = """""The U.S. said it would grant a temporary license from Oct. 21 through to April next year to allow businesses to manufacture some of the high-tech products in China for use outside the country."
aliens explain the meaning of this sentence as follows:
The U.S. said it would grant a temporary license| The license would be granted from Oct. 21 through to April next year| The license would allow businesses to manufacture some of the high-tech products in China for use outside the country|
"The program Fritz falls short of superintelligence—even though it is much better than humans at chess—because Fritz cannot outperform humans in other tasks."
aliens explain the meaning of this sentence as follows:
The program Fritz falls short of superintelligence| Even though it is much better than humans at chess, Fritz cannot outperform humans in other tasks|
"%Q"
aliens explain the meaning of this sentence as follows:
"""
        
        promptText1 = promptText.replace("%Q", argText)

        #inputText = """"The U.S. said it"""
        nOut = 300
        print("inference ({+300})...")
        responseTextRaw = generator(promptText1, max_length=(len(promptText1)+nOut))[0]["generated_text"]
        print("...done")
        #print(responseTextRaw) DBG

        responseText0 = responseTextRaw[len(promptText1):]
        
        idxNewLine = responseText0.find("\n")
        responseText1 = responseText0[:idxNewLine] # cut away everthing after the new line which represents the end of the solution
        
        arr0 = responseText1.split("|")
        arr1 = arr0[:len(arr0)-1] # remove last item which is empty
        return arr1

    
    

# script to parse a simple part of a sentence into relations where it is easy to extract information
# DOESNT work always!!!
class ProcessTextToSimple1(object):
    def __init__(self):
        pass
    
    def process(self,text,tokenizer_flan,model_flan):
        # tested with flan-t5-xl
        input_text = """A:"We like you"
B:like(We,you)
A:"Parrots are black and yellow"
B:parrots(color):black,yellow
A:"pressberry is great"
B:pressberry(software):great
A:"it is a great place to find or post information"
B:rating(place):great;action():post
A:"Tim loves alex"
B:tim(love):alex
A:"Sites are all sorted by traffic popularity"
B:sites(popularity):traffic
A:"%Q"
B:""".replace("%Q",text)

        input_ids = tokenizer_flan(input_text, return_tensors="pt").input_ids

        outputs = model_flan.generate(input_ids, max_new_tokens=100)
        resText=tokenizer_flan.decode(outputs[0])
  
        #print(f"{resText}") # DBG
        return resText


    
    
# script to parse a simple part of a sentence into relations where it is easy to extract information
#
# tries to convert it to a relationship form which is easy to parse
class ProcessTextToSimple3(object):
    def __init__(self):
        pass
    
    def process(self,generator,argText):
        # tested with bloom-176b
        # ref https://gist.github.com/PtrMan/a4f0ffa734b6df6d0d16387f82e42303#file-0045-tool-convert-simpler-english-to-relation-txt-L1
        promptText = """This tool converts the sentence "the cat is fat" into a formal definition. cat IS fat
This tool converts the sentence "Professors and lecturers should at least discuss the effects of the new tools from OpenAI, Microsoft and DeepMind" into a formal definition. UNKNOWN
This tool converts the sentence "the boilingpoint of T555 is 55 degree celsius" into a formal definition. boilingpoint of T555 IS 55 degree celsius
This tool converts the sentence "%Q" into a formal definition."""
        promptText1 = promptText.replace("%Q", argText)

        nOut = 130
        print("inference ({+130})...")
        responseTextRaw = generator(promptText1, max_length=(len(promptText1)+nOut))[0]["generated_text"]
        print("...done")
        #print(responseTextRaw) DBG

        responseText0 = responseTextRaw[len(promptText1):]
        
        
        return responseText0

    
    
# prompt to pose factual knowledge as a question to answer it.
#
#
class ProcessTextToQa3(object):
    def __init__(self):
        self.verbose = False
    
    def process(self,generator,argText):
        text0, isComplete = self._processRetText(generator,argText)
        lines0 = text0.split("\n")
        if not isComplete:
            lines0 = lines0[:len(lines0)-1] # remove last line because last line is incomplete
        lines0 = list(set(lines0)) # remove duplicates
        return lines0
    
    def _processRetText(self,generator,argText):
        # tested with bloom-176b
        # ref https://gist.github.com/PtrMan/a4f0ffa734b6df6d0d16387f82e42303#file-0046-tool-fact-extraction-with-self-supervised-qa-a-txt-L1
        promptText = """BEGIN
Its density is 19.30 grams per cubic centimetre (0.697 lb/cu in), comparable with that of uranium and gold, and much higher (about 1.7 times) than that of lead.
We come up with the following questions:
What is its density? Its density is 19.30 grams per cubic centimetre.
the density of it is comparable with that of uranium and gold? YES
its density is much higher than that of lead? YES
BEGIN
Polycrystalline tungsten is an intrinsically brittle and hard material (under standard conditions, when uncombined), making it difficult to work.
We come up with the following questions:
Polycrystalline tungsten is an intrinsically brittle and hard material? YES
Polycrystalline tungsten is difficult to work? YES
BEGIN
%Q
We come up with the following questions:"""
        promptText1 = promptText.replace("%Q", argText)

        
        
        
        nOutIncrement = 7
        
        responseTextRawAccu = ""
        for iRound in range(int(120/nOutIncrement)):
            if self.verbose:
                print(f"inference (+{nOutIncrement})...")
            
            nTokens = len(generator.tokenizer(promptText1+responseTextRawAccu)["input_ids"]) # we need to know how many tokens we have
            
            responseTextRaw = generator(promptText1+responseTextRawAccu, max_length=nTokens+nOutIncrement, return_full_text=False)[0]["generated_text"]

            
            responseTextRawAccu += responseTextRaw
            if responseTextRawAccu.find("\nBEGIN")!=-1:
                break
            
            
            #print(responseTextRaw) DBG
        
        if self.verbose:
            print("...done")
        
        responseText0 = responseTextRawAccu
        
        idxEnd = responseText0.find("\nBEGIN")
        isComplete = idxEnd!=-1
        responseText1 = responseText0[:idxEnd] # cut away everthing after the new line which represents the end of the solution
        
        return (responseText1, isComplete)
