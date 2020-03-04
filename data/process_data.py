import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    导入message、categories文件
    返回合并数据集
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath) 
    
    df = pd.merge(messages, categories, on = 'id')
    
    return df

def clean_data(df):
    """
    数据集清洗并返回
    """
    categories = df['categories'].str.split(';', expand = True)
    
    #选择categories的首行
    row = categories.iloc[0]
    #删去每个字符串的后两位作为列名
    category_colnames = row.apply(lambda x: x[0:-2])
    #重命名categories的列名
    categories.columns = category_colnames
    
    for column in categories:
        #修改值为字符串最后一位，例如1，0
        categories[column] = categories[column].str[-1]
        #修改数据类型
        categories[column] = categories[column].astype('int64')
        #检查数值，将数值全部转化为0或1
        categories[column] = categories[column].apply(lambda x:1 if x > 0 else 0)

    #删除原有的categories列
    df.drop('categories', axis = 1, inplace = True)
    #使用concat合并数据集
    df = pd.concat([df, categories], axis = 1)
    #删除重复行
    df.drop_duplicates(inplace=True)
    
    return df
    
def save_data(df, database_filename):
    """
    保存清洗后的数据
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Messages', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
