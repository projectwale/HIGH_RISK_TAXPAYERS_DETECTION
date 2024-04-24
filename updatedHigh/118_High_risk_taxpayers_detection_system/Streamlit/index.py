import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from matplotlib import pyplot as plt
#%matplotlib inline
import matplotlib
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

st.write("""
# Cross site scripting Detection 

Built with streamlit
""")

menu  = ["EDA", "Naive bayes", "SVM","KNN", "Prediction"]
st.sidebar.title("Navigation")
choices = st.sidebar.selectbox("Select Activities", menu)


def main():
    data = pd.read_csv((r'C:\Users\sushant\Desktop\project\Realtime_xss_detection\Flask_streamlit\Streamlit\data\XSS_dataset.csv'), encoding='ISO-8859-1')
    df = data

    if choices == 'EDA':
        st.subheader("EDA")
        if st.sidebar.checkbox("show summary"):
            st.write(df.describe())
        
        if st.sidebar.checkbox("show shape"):
            st.write(df.shape)
        if st.sidebar.checkbox("Bar chart"):
            st.write(df.Label.value_counts().plot(kind = 'bar', stacked = True, figsize = (12, 8), title = 'Distribution of Rating among apps available on Google Play Store')) # Historgram of frequencies 
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.xlabel('Sentence')
            plt.ylabel('Frequencies')
            plt.show()
            st.pyplot()
            plt.show()		
        if st.sidebar.checkbox("Pie Chart"):
            st.write(df['Sentence'].value_counts().plot.pie(autopct="%1.1f%%"))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            plt.show()
        if st.sidebar.checkbox("Word Image"):
            #df.drop_duplicates(inplace = True)
            df.drop('sr', axis = 'columns', inplace = True)
            import nltk
            #nltk.download('all')
            # Step - a : Remove blank rows if any.
            df['Sentence'].dropna()
            # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
            df['Sentence'] = [str(entry).lower() for entry in df['Sentence']]
            # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
            df['Sentence']= [word_tokenize(entry) for entry in df['Sentence']]
            # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
            # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
            tag_map = defaultdict(lambda : wn.NOUN)
            tag_map['J'] = wn.ADJ
            tag_map['V'] = wn.VERB
            tag_map['R'] = wn.ADV

            for index,entry in enumerate(df['Sentence']):
                # Declaring Empty List to store the words that follow the rules for this step
                Final_words1 = []
                # Initializing WordNetLemmatizer()
                word_Lemmatized = WordNetLemmatizer()
                # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
                for word, tag in pos_tag(entry):
                    # Below condition is to check for Stop words and consider only alphabets
                    if word not in stopwords.words('english') and word.isalpha():
                        word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                        Final_words1.append(word_Final)
                # The final processed set of words for each iteration will be stored in 'text_final'
                df.loc[index,'final_text'] = str(Final_words1)
            text = " ".join(i for i in df['final_text'])
            wordcloud = WordCloud(max_font_size = 50, background_color = "white").generate(text)
            plt.figure(figsize = [10,10])
            plt.imshow(wordcloud, interpolation = "bilinear")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            plt.show()
            plt.show()
    
    if choices=="Naive bayes" :
        data = pd.read_csv(r'C:\Users\sushant\Desktop\project\Realtime_xss_detection\Flask_streamlit\Streamlit\data\XSS_dataset.csv',encoding='latin-1',nrows=1000,error_bad_lines=False)
        data = df

        #df.drop_duplicates(inplace = True)
        df.drop('sr', axis = 'columns', inplace = True)
        import nltk
        #nltk.download('all')
        # Step - a : Remove blank rows if any.
        df['Sentence'].dropna()
        # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
        df['Sentence'] = [str(entry).lower() for entry in df['Sentence']]
        # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
        df['Sentence']= [word_tokenize(entry) for entry in df['Sentence']]
        # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
        # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        for index,entry in enumerate(df['Sentence']):
            # Declaring Empty List to store the words that follow the rules for this step
            Final_words1 = []
            # Initializing WordNetLemmatizer()
            word_Lemmatized = WordNetLemmatizer()
            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            for word, tag in pos_tag(entry):
                # Below condition is to check for Stop words and consider only alphabets
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                    Final_words1.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
            df.loc[index,'final_text'] = str(Final_words1)
        df1 = df.replace(r'</,^\s*$@ð#<>?!%^&*', np.nan, regex=True)
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df1['final_text'],df1['Label'],test_size=0.3)
        from sklearn.feature_extraction.text import TfidfVectorizer
        Tfidf_vect = TfidfVectorizer(max_features=5000, sublinear_tf=True, decode_error='ignore')
        #Tfidf_vect = TfidfVectorizer(max_features=5000)
        Tfidf_vect.fit(df1['final_text'])
        Train_X_Tfidf = Tfidf_vect.transform(Train_X)
        Test_X_Tfidf = Tfidf_vect.transform(Test_X)
        if st.sidebar.checkbox('naive bayes accuracy'):
            Naive = naive_bayes.MultinomialNB()
            Naive.fit(Train_X_Tfidf,Train_Y)
            predictions_NB = Naive.predict(Test_X_Tfidf)
            st.write("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
            
        
        if st.sidebar.checkbox('naive bayes precision score, recall and f1 Score'):   
            from sklearn.metrics import precision_recall_fscore_support
            op=precision_recall_fscore_support(Test_Y, predictions_NB, average='macro')

            st.write("precision is "+str(op[0]))
            st.write("recall is "+str(op[1]))
            st.write("f1score is "+str(op[2]))



        if st.sidebar.checkbox('naive bayes heatmap'): 
            # fit the training dataset on the NB classifier
            Naive = naive_bayes.MultinomialNB()
            Naive.fit(Train_X_Tfidf,Train_Y)
            # predict the labels on validation dataset
            predictions_NB = Naive.predict(Test_X_Tfidf)
            expected =Encoder.inverse_transform(Test_Y.tolist()) 
            predicted =Encoder.inverse_transform(predictions_NB.tolist())    
            labels = listofclasses # ['delhi.txt', 'mumbai.txt','pune.txt']
            ax= plt.subplot()
            cm = confusion_matrix(expected, predicted, labels)
            st.write(sns.heatmap(cm, annot=True, ax = ax)) #annot=True to annotate cells # labels, title and ticks
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels') 
            ax.set_title('Confusion Matrix') 
            ax.xaxis.set_ticklabels(labels)
            ax.yaxis.set_ticklabels(labels)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()	

    if choices=="SVM" :
        data = pd.read_csv(r'C:\Users\sushant\Desktop\project\Realtime_xss_detection\Flask_streamlit\Streamlit\data\XSS_dataset.csv',encoding='latin-1',nrows=1000,error_bad_lines=False)
        data = df

        #df.drop_duplicates(inplace = True)
        df.drop('sr', axis = 'columns', inplace = True)
        import nltk
        #nltk.download('all')
        # Step - a : Remove blank rows if any.
        df['Sentence'].dropna()
        # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
        df['Sentence'] = [str(entry).lower() for entry in df['Sentence']]
        # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
        df['Sentence']= [word_tokenize(entry) for entry in df['Sentence']]
        # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
        # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        for index,entry in enumerate(df['Sentence']):
            # Declaring Empty List to store the words that follow the rules for this step
            Final_words1 = []
            # Initializing WordNetLemmatizer()
            word_Lemmatized = WordNetLemmatizer()
            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            for word, tag in pos_tag(entry):
                # Below condition is to check for Stop words and consider only alphabets
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                    Final_words1.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
            df.loc[index,'final_text'] = str(Final_words1)
        df1 = df.replace(r'</,^\s*$@ð#<>?!%^&*', np.nan, regex=True)
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df1['final_text'],df1['Label'],test_size=0.3)
        from sklearn.feature_extraction.text import TfidfVectorizer
        Tfidf_vect = TfidfVectorizer(max_features=5000, sublinear_tf=True, decode_error='ignore')
        #Tfidf_vect = TfidfVectorizer(max_features=5000)
        Tfidf_vect.fit(df1['final_text'])
        Train_X_Tfidf = Tfidf_vect.transform(Train_X)
        Test_X_Tfidf = Tfidf_vect.transform(Test_X)
        
        if st.sidebar.checkbox('SVM accuracy'): 
            Tfidf_vect = TfidfVectorizer(max_features=5000)
            Tfidf_vect.fit(Corpus['text_final'])
            Train_X_Tfidf = Tfidf_vect.transform(Train_X)
            Test_X_Tfidf = Tfidf_vect.transform(Test_X)
            #Import svm model
            from sklearn import svm
            #Create a svm Classifier
            clf = svm.SVC(kernel='linear') # Linear Kernel
            #Train the model using the training sets
            clf.fit(Train_X_Tfidf, Train_Y)
            #Predict the response for test dataset
            y_pred = clf.predict(Test_X_Tfidf)
            st.write("SVM Accuracy Score -> ",accuracy_score(y_pred, Test_Y)*100)
        
    if choices=="KNN" :
        data = pd.read_csv(r'C:\Users\sushant\Desktop\project\Realtime_xss_detection\Flask_streamlit\Streamlit\data\XSS_dataset.csv',encoding='latin-1',nrows=1000,error_bad_lines=False)
        data = df

        #df.drop_duplicates(inplace = True)
        df.drop('sr', axis = 'columns', inplace = True)
        import nltk
        #nltk.download('all')
        # Step - a : Remove blank rows if any.
        df['Sentence'].dropna()
        # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
        df['Sentence'] = [str(entry).lower() for entry in df['Sentence']]
        # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
        df['Sentence']= [word_tokenize(entry) for entry in df['Sentence']]
        # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
        # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        for index,entry in enumerate(df['Sentence']):
            # Declaring Empty List to store the words that follow the rules for this step
            Final_words1 = []
            # Initializing WordNetLemmatizer()
            word_Lemmatized = WordNetLemmatizer()
            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            for word, tag in pos_tag(entry):
                # Below condition is to check for Stop words and consider only alphabets
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                    Final_words1.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
            df.loc[index,'final_text'] = str(Final_words1)
        df1 = df.replace(r'</,^\s*$@ð#<>?!%^&*', np.nan, regex=True)
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df1['final_text'],df1['Label'],test_size=0.3)
        from sklearn.feature_extraction.text import TfidfVectorizer
        Tfidf_vect = TfidfVectorizer(max_features=5000, sublinear_tf=True, decode_error='ignore')
        #Tfidf_vect = TfidfVectorizer(max_features=5000)
        Tfidf_vect.fit(df1['final_text'])
        Train_X_Tfidf = Tfidf_vect.transform(Train_X)
        Test_X_Tfidf = Tfidf_vect.transform(Test_X)
            
        if st.sidebar.checkbox('SVM accuracy'): 
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=3)
            # Train the model using the training sets
            model.fit(Train_X_Tfidf, Train_Y)
            #Predict Output
            predicted= model.predict(Test_X_Tfidf) # 0:Overcast, 2:Mild
            st.write("KNN Accuracy Score -> ",accuracy_score(predicted, Test_Y)*100)
    
    if choices=="Prediction":
        data = pd.read_csv(r'C:\Users\sushant\Desktop\project\Realtime_xss_detection\Flask_streamlit\Streamlit\data\XSS_dataset.csv',encoding='latin-1',nrows=1000,error_bad_lines=False)
        data = df

        #df.drop_duplicates(inplace = True)
        df.drop('sr', axis = 'columns', inplace = True)
        import nltk
        #nltk.download('all')
        # Step - a : Remove blank rows if any.
        df['Sentence'].dropna()
        # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
        df['Sentence'] = [str(entry).lower() for entry in df['Sentence']]
        # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
        df['Sentence']= [word_tokenize(entry) for entry in df['Sentence']]
        # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
        # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        for index,entry in enumerate(df['Sentence']):
            # Declaring Empty List to store the words that follow the rules for this step
            Final_words1 = []
            # Initializing WordNetLemmatizer()
            word_Lemmatized = WordNetLemmatizer()
            # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
            for word, tag in pos_tag(entry):
                # Below condition is to check for Stop words and consider only alphabets
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                    Final_words1.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
            df.loc[index,'final_text'] = str(Final_words1)
        df1 = df.replace(r'</,^\s*$@ð#<>?!%^&*', np.nan, regex=True)
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df1['final_text'],df1['Label'],test_size=0.3)
        
        i = st.text_input("Enter your link")
        from sklearn.feature_extraction.text import TfidfVectorizer
        Tfidf_vect = TfidfVectorizer(max_features=5000, sublinear_tf=True, encoding='utf-8', decode_error='ignore')
        #Tfidf_vect = TfidfVectorizer(max_features=5000)
        Tfidf_vect.fit(df1['final_text'])
        #Train_X_Tfidf = Tfidf_vect.fit_transform(Train_X)
        #Test_X_Tfidf = Tfidf_vect.transform(Test_X)
        
        tf2=Tfidf_vect.transform(i) 

        if st.button("It is XSS!"):
            
            #naive bayes
            with open(r'C:\Users\sushant\Desktop\project\Realtime_xss_detection\Flask_streamlit\Streamlit\model\nbs_pickle.sav','rb') as f:
                mp=pickle.load(f)
                red=mp.predict(tf2)
            if red==0 :
                st.write("XSS attack not detected by Naive Bayes")
            elif red==1:
                st.write("XSS attack detected by Naive Bayes")
            else:
                st.write("Somthing went wrong detected by Naive Bayes")
    
            #SVM    
            with open(r'C:\Users\sushant\Desktop\project\Realtime_xss_detection\Flask_streamlit\Streamlit\model\svm_pickles.sav','rb') as f:
                mp=pickle.load(f)
                red=mp.predict(tf2)
            if red==0 :
                st.write("XSS attack not detected by SVM")
            elif red==1:
                st.write("XSS attack detected by SVM")
            else:
                st.write("Somthing went wrong detected by SVM")
            #KNN
            with open(r'C:\Users\sushant\Desktop\project\Realtime_xss_detection\Flask_streamlit\Streamlit\model\kn_pickle.sav','rb') as f:
                mp=pickle.load(f)
                red=mp.predict(tf2)
            if red==0 :
                st.write("XSS attack not detected by random forest")
            elif red==1:
                st.write("XSS attack detected by random forest")
            else:
                st.write("Somthing went wrong detected by random forest")
          

        
if __name__ == '__main__':
	main()
