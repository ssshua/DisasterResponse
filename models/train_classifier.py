import sys
import pickle
import pandas as pd
import re
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    从SQLLite导入数据
    返回:X, y, category_names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Messages', engine)
    
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    return X, Y, Y.columns
    
def tokenize(text):
    """
    分词，删除停顿词，文本处理
    """
    
    #删除标点符号
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    #导入英文停顿词库
    stopword = stopwords.words('english')
      
    clean_tokens = []
    for tok in tokens:
        #词形还原
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        #去除停顿词
        if tok in stopword:
            pass
        else:
            clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """
    构建模型的管道函数
    返回模型
    """
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    #构建参数词典
    parameters = {
        'vect__ngram_range': ((1, 1), (1,2)),
        'clf__estimator__n_estimators' : [50, 100]
        }
    
    #设置网格搜索
    cv = GridSearchCV(pipeline, parameters, cv = 2, n_jobs = -1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    输出模型评价
    Model:构建的模型
    X_test:测试集
    Y_test:测试集分类
    category_names
    """
    
    y_pred = model.predict(X_test)
    
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    模型保存为pickle
    """
    pkl = open(model_filepath, 'wb')
    pickle.dump(model, pkl)
    pkl.close


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

