import yake
import re
    
# Input Text
text = '''Creating appropriate IT&C infrastructure in NHPC to support business functions and organizational growth. Developing a policy framework for procurement, issuance, maintenance and usage of IT&C infrastructure in NHPC in a cost effective, environment friendly and sustainable manner. Establishing policies for providing IT&C services with utmost concern for information security and legal provisions.'''

# Specifying Parameters
language = "en"
max_ngram_size = 3
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 100
pattern = r"'(.*?)'"
all_ner_words = ""

custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(text)

#print(keywords)
for kw in keywords:
    extracted_text = re.findall(pattern, str(kw))
    #temp = extracted_text[0]
    all_ner_words = all_ner_words+(extracted_text[0])+","
    
print(all_ner_words)


