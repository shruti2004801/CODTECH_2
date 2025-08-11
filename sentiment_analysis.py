import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
# read dataset
df = pd.read_csv('twitter_sentiment.csv',encoding='latin1')
# use encoding="latin1" to specify the special characters in the data set
print(df)
#check null
print(df.isnull().sum())

# Convert categorical to numerical
df['sentiment']=df['sentiment'].replace({'positive':1, 'negative':0})
print(df)

# Feature separation
x=df.drop(['id','sentiment'],axis=1)
x=df['text']
y=df['sentiment']
print(x)
print(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
print(" x training data:",x_train)
print("x testing data",x_test)
print("y training data",y_train)
print("y testing data",y_test)

#Tf-idf vectorizer
tfv=TfidfVectorizer()
x_train_tfidf=tfv.fit_transform(x_train)
x_test_tfidf=tfv.transform(x_test)
print("training data",x_train_tfidf)
print("testing data",x_test_tfidf)

#logesticRegrassion
logr=LogisticRegression()
logr.fit(x_train_tfidf,y_train)

#prediction score
y_predict=logr.predict(x_test_tfidf)
print(y_predict)

#Accuracy score
accuracy=accuracy_score(y_test,y_predict)
print("Accuracy score:",accuracy)
print("Classification Report:",classification_report(y_test,y_predict))