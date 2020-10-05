"""
This will serve as my notes for cleaning and creating the environments that I need

step 1:
    Create pipenv virtual environment and download all maing packages
        pipenv install numpy pandas spacy tensorflow TextBlob scikit-learn
        pipenv run python3 -m spacy download en_core_web_md
        pipenv run python3 -m textblob.download_corpora 

step 2:
    Data Exploration
        using pandas to read the static or flat file, use read_csv and encoding='latin1'
       
        Set Columns:
            slice the df = df[[0,5]]
            df.columns = ['name1','name2']

        Look at unique value counts for target column:
            df['target_col'].value_counts()
        
        Look for:
            Word Count:
            Stop Word Count
            Char Count
            Punctuation Count
            Avrg Word Count

Step 3:
    Data Cleaning
        Turn everyting to lower
        Correct Spelling
        Remove all emails
        Remove all @ first
        Remove all hashtags
        Remove and strip all <foo>
        Remove all HTTPS website protocols -> Further work is needed for this
        Clean and Replace all contrractions
        remove all numerical values
        remove all special characters (punctuations as well)       
        Remove all super common words
        Remove all rare words
        remove all stop words

Step 4:
    Word Embeddings
        Pick which word embeddings you will use (GloVe, Spacy, NLTK, etc..)
        Download the txt file locally and import into project
        Create an empty dictionnary which will contain our word : x_int dimension representation
        Typically:
            iterate through every line in the embedding file
            split the line into tokens
            word is value[0]
            vector is np.asarray(value[1:])
            dictionnary[word] = vector 
            close file

Step 4:
    Model Preperation
        To prepare to feed into our model we need to make sure our data is properly configured
        Convert from series to a list
        Declare Y (Label/Target)
        Declare the tokenizer
        Declare vocab size
        encode the text with the tokenizer
        Pad the sequences
        
    Declare our word matrix 
        Assign our vector matrix to be of the same dimensions as our embedding layer
        Check if the word exists in our embedding dictionnary 

Step 5:
    Model Building
        Split our Data
        Create and assign out layes
        Compile the model
        Fit the model

"""
###Code Snippets

## Data Exploration // Get More Features

#Get word counts
x = x.apply(lambda x: len(str(x).split()))
#Get char counts
x = x.apply(lambda x: len(str(x)))
#Get Stop Word Count
x = x.apply(lambda x: len([word for word in x.split() if word in stopwords]))
#Get punctuation count
x = x.apply(lambda x: len([c for c in x.split() if c in string.punctuation]))
#Get count if words starts with specific character 'x'
x = x.apply(lambda x: len([word for word in x.split() if word.startswith('x')]))
#Get all URLS
urls = re.findall(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)
#Get all Emails
email = re.findall(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+\b)', text)

#** Additional
def avrg_word_len (x):
    words = x.split()
    word_len = 0
    for word in words:
        word_len = word_len + len(word)
    return word_len/len(words)  


## Data Cleaning

#Turn text lower
text = text.lower()
#Find and replace all emails:
text = re.sub("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", '', text)
# remove all @ first
text = re.sub(r'@([A-Za-z0-9_]+)', "", text)
# # remove and strip all retweets (RT)
text = re.sub(r'\brt:\b', '', text).strip()
# find and replace all websites
text = re.sub(
    r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', text)
# find and replace all non-alpha numerical valu
text = re.sub(r'[^A-Z a-z]+', '', text)
# #Remove accented characters
text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
# # fix all potential spelling mistakes
text = str(TextBlob(text).correct())
# remove all numerical values
x = re.sub(r'[0-9]+', "", x)
# remove all special characters
x = re.sub(r'[^\w ]+', ' ', x)
# We are removed all the workds that are in our top 10 **
x = [words for words in x if words not in Top_10]
# # We are rempoving all the words that are not in our rare list *8
x = [words for words in x if words not in rare_words]
# remove all the words in our STOP_WORDS **
x = [words for words in x if words not in STOP_WORDS]
# # clean and replace with contractions **
x = contractions_replace(x)
# #Make base form of words AKA Lemmatize w/SPACY **
x = get_base_lemma(x)

#*** additional needs
text = ' '.join(df['column'])
text = text.split()
freq_ = pd.Series(text).value_counts()
Top_10 = freq_[:10]
Least_freq = freq_[freq_.values == 1]

def contractions_replace(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    return x

def get_base_lemma(x):
    x = str(x)
    x_list = []
    doc = nlp(x)
    
    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text
        x_list.append(lemma)

    return ' '.join(x_list)


## Word Embeddings

embedding_dict = dict()
file = open('location',encoding='utf-8')
#Create the word embeddings
for line in file:
    value = line.split()
    word = value[0]
    vector = np.asarray(value[1:])
    embedding_dict[word] = vector
file.close()

#Save using pickle
with open('test_file.pickle','wb') as handle:
    pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



## Model Preperation

#Conver to list 
text = df['twitts'].tolist()
#Create target
y = df['sentiment']
#Declare Token
token = Tokenizer()
token.fit_on_texts(text)
#Create Vocab length
vocab_size = len(token.word_index) + 1
#Encode our text
encoded_text = token.texts_to_sequences(text)
# Pad the sequences
max_len = max([len(s.split()) for s in text])
#Delcare X
X = pad_sequences(encoded_text, maxlen=max_len, padding='post')

#Open our embedding saved file
with open('test_file.pickle', 'rb') as handle:
    data_test = pickle.load(handle)
# create empty matrix with the proper size
word_vector_matrix = np.zeros((vocab_size, 200))

for word, index in token.word_index.items():
    vector = data_test.get(word)
    # check if the word is not present in GloVe
    if vector is not None:
        word_vector_matrix[index] = vector
    else:
        print(word)


## Model Build This is for Embedding LSTM
x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y)
#The vector size has to match the size of the dimensions in our embedding layer
vec_size = 200

#Model can be changed absolutely, layers added, etc, etc..
model = tf.keras.Sequential()
model.add(Embedding(vocab_size, vec_size, input_length=max_len,
                    weights=[word_vector_matrix], trainable=False))
model.add(Conv1D(64, 8, activation="relu"))
model.add(MaxPooling1D(2))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))





##Contractions list:
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how does",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    " u ": " you ",
    " ur ": " your ",
    " n ": " and ",
    "won't": "would not",
    "dis": "this",
    "brng": "bring"
}




