# Question-1 - Exploring NLTK
import nltk
nltk.download()
from nltk.book import *
text1.concordance("monstrous")
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
sorted(set(text3))
len(set(text3))

from __future__ import division
len(text3) / len(set(text3))

text3.count("smote")

100 * text4.count('a') / len(text4)

# Question-2 - Regular Expressions
import nltk
print ("\nRegular Expressions\n")

print ("\n[a-zA-Z]+, It will match one or more uppercase and lowercase ascii letters\n")
t1 = "This book contains information obtained from authentic and highly regarded sources."
nltk.re_show('[a-zA-Z]+', t1)

print("\n[A-Z][a-z]*, It will match zero or more ascii letters, due to use of '*'\n")
t2 = "This book contains information obtained from authentic and highly regarded sources."
nltk.re_show('[A-Z][a-z]*', t2)

print("\np[aeiou]{,2}t, It will match 'p' followed by 2 vowels and a 't'\n")
t3 = "You are the real phantom of arabia, living in tunisia"
nltk.re_show('p[aeiou]{,2}t', t3)

print("\n\d+(\.\d+)?, It will match currency and percentages, e.g. $12.40, 82%, after a dot as well\n")
t4 = "That U.S.A. poster-print costs $12.40..."
nltk.re_show('\n\d+(\.\d+)?', t4)

print("\n([^aeiou][aeiou][^aeiou])*, It will match a combination of consonant, vowel and consonant\n")
t5 = "abcdefghijklmnopqrstuvwxyz"
nltk.re_show('([^aeiou][aeiou][^aeiou])*', t5)

print("\n\w+|[^\w\s]+, It will match one or more spaces, non-letters, letters\n")
t6 = "abcdefgh ijklmnopqrs tuvwxyz 65"
nltk.re_show('\w+|[^\w\s]+', t6)

# Question-3 - Write regular expressions to match the following classes of strings
# Part a - A single determiner (assume that a, an, and the are the only determiners).
import re
text = nltk.Text(word.lower() for word in nltk.corpus.words.words('en'))
for t in text:
    if re.search(r'^(an?|the)$', t):
        print(t)

# Part b - An arithmetic expression using integers, addition, and multiplication, such as 2*3+8.
text = 'An arithmetic expression using integers, addition, and multiplication, such as 2*3+8'
nltk.re_show(r'8', text)
nltk.re_show(r'[\d*]', text)
nltk.re_show(r'[\d+*]', text)
nltk.re_show(r'[\d+*-/]', text)

# Question-4 - 
# Part a - Write a utility function that takes a URL as its argument, 
# and returns the contents of the URL
import re
from urllib import request
from bs4 import BeautifulSoup
def utilityfunc(url):
    data = request.urlopen(url).read().decode('utf-8');
    data = re.sub(r'\n', ' ', data)
    data = re.sub(r'<script.*?script>', '', data, re.S)
    data = re.sub(r'<.*?>', '', data)
    data = re.sub(r'\s+', ' ', data)
    
    return data
html = 'https://www.csail.mit.edu/people?person%5B0%5D=role%3A299'
# Check how the original data looks like
html_doc = request.urlopen(html).read().decode('utf8')
print(html_doc)
# After cleaning the data by the utility function
final_doc = utilityfunc(html)
print(final_doc)

# Part b - Parse data using BeautifulSoup
soup = BeautifulSoup(html_doc, 'html.parser')
print(soup.prettify())
print(soup.get_text())

# Question-5 - Tokenize text parsed from the above url using nltk. 
# Find all phone numbers and email addresses from this text using regular expressions.
import re 
# All the emails from the above text
email = re.findall('\S+@\S+', final_doc)
print(email) 
# All the phone numbers
phone = re.findall('\([0-9](3)\)-[0-9](3)-[0-9](4)', final_doc)
print(phone) 

# Question-6 - Use the Porter Stemmer to normalize some tokenized text, calling the stemmer on each word.
# Do the same thing with the Lancaster Stemmer and see if you observe any differences
import nltk
from nltk.corpus import abc
text = abc.words()

porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()

for w in text:
    print(w)
    # Word after implementing Porter Stemmer
    print(porter.stem(w))
    # Word after implementing Lancaster Stemmer
    print(lancaster.stem(w))