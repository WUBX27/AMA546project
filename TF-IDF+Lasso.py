import json
import glob
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

from scipy.sparse import csr_matrix, hstack

# TF-IDF+Lasso
# --------------------------------------------------
# 读取并解析JSON文件
def load_json_data(file_path):
    """加载单个JSON文件并提取需要的数据"""
    texts = []
    logvol_minus = []
    logvol_plus = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["tok"])                   # 提取文本
            logvol_minus.append(data["logvol-12"])      # 提取logvol-12
            logvol_plus.append(data["logvol+12"])       # 提取logvol+12
            
    return texts, logvol_minus, logvol_plus


def get_json_paths(years):
    """
    根据年份列表生成对应的JSON文件路径
    
    参数：
    years     - list, 年份列表，支持int或str类型（如[1996, 2000]或["1996","2000"]）
    
    返回：
    paths     - list，包含完整路径的JSON文件路径列表
    """
    paths = []
    for year in years:
        # 统一转换为字符串格式，处理可能存在的整数输入
        year_str = str(year).strip()  # 去除前后空格
        
        # 构建文件名：年份.json
        filename = f"{year_str}.json"
        
        #获取当前json文件路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        
        paths.append(file_path)
    return paths


def load_jsonfiles(paths):
    """
    逐个读取JSON文件并提取内容
    :param paths: 需要读取的文件路径列表
    :return: 包含所有文档内容的列表
    """
    texts = []
    logvol_minus = []
    logvol_plus = []
    
    for file_path in paths:
        texts_temp = []
        logvol_minus_temp = []
        logvol_plus_temp = []
        #提取json文件中的数据
        texts_temp,logvol_minus_temp,logvol_plus_temp = load_json_data(file_path)
        texts = texts + texts_temp 
        logvol_minus = logvol_minus + logvol_minus_temp 
        logvol_plus= logvol_plus + logvol_plus_temp
    return texts, logvol_minus, logvol_plus


def prepare_data(train_years, test_years):
    """
    根据训练集和测试集的年份列表，生成特征矩阵和目标变量
    
    参数：
    train_years - list, 训练集年份列表
    test_years  - list, 测试集年份列表
    
    返回：
    X_train, X_test, y_train, y_test
    
    """
    
    # = 1. 生成文件路径 = 
    train_filepaths = get_json_paths(train_years)
    test_filepaths = get_json_paths(test_years)

    # = 2. 加载数据 = 
    
    # 加载训练集
    train_texts,train_logvol_minus,train_logvol_plus=load_jsonfiles(train_filepaths)
    # 加载测试集 
    test_texts, test_logvol_minus,test_logvol_plus = load_jsonfiles(test_filepaths)
    
    
    # = 3. 文本特征提取 = 
    vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),      # 增加二元语法
    min_df=5,                # 忽略出现少于5次的词
    max_df=0.9               # 忽略超过90%文档的词
)
    train_tfidf = vectorizer.fit_transform(train_texts)  # 只在训练集fit
    test_tfidf = vectorizer.transform(test_texts)        # 测试集用相同vectorizer

    # = 4. 合并数值特征 = 
    def merge_features(tfidf_matrix, logvol_values):
        """合并TF-IDF矩阵和logvol-12特征"""
        return hstack([
            tfidf_matrix,
            csr_matrix(logvol_values).reshape(-1, 1)  # 确保形状为(n_samples, 1)
        ], format='csr')

    X_train = merge_features(train_tfidf, train_logvol_minus)
    X_test = merge_features(test_tfidf, test_logvol_minus)

    # 转换目标变量为numpy数组
    y_train = np.array(train_logvol_plus, dtype=np.float32)
    y_test = np.array(test_logvol_plus, dtype=np.float32)

    print(f"训练集形状：{X_train.shape}，测试集形状：{X_test.shape}")

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    
    predictions = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f'测试结果:')
    print(f'  MSE: {mse:.4f}')
    print(f'  RMSE: {rmse:.4f}')
    print(f'  R²: {r2:.4f}')
    
    return mse, rmse, r2, predictions

def model_train(model,train_years, test_years):
    X_train, X_test, y_train, y_test=prepare_data(train_years, test_years)
    model.fit(X_train,y_train)

    train_score=model.score(X_train,y_train)
    test_score=model.score(X_test,y_test)

    coeff_used = np.sum(model.coef_!=0)
    print ("training score : ", train_score )
    print ("test score     : ", test_score)
    print ("number of features used: ", coeff_used)

    mse, rmse, r2, pred=evaluate_model(model, X_test, y_test)
    return mse, rmse, r2, pred

def load_files(file_list):
    """辅助函数：批量加载文件数据"""
    logvol_minus, logvol_plus = [], []
    for file in file_list:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                logvol_minus.append(data["logvol-12"])
                logvol_plus.append(data["logvol+12"])
    return logvol_minus, logvol_plus

def prepare_data_LO(train_years, test_years):
    """
    根据训练集和测试集的年份列表，生成特征矩阵和目标变量
    
    参数：
    train_years - list, 训练集年份列表
    test_years  - list, 测试集年份列表
    
    返回：
    X_train, X_test, y_train, y_test
    
    """
    
    # = 1. 生成文件路径 = 
    train_filepaths = get_json_paths(train_years)
    test_filepaths = get_json_paths(test_years)

    # = 2. 加载数据 = 
    
    # 加载训练集
    train_logvol_minus,train_logvol_plus=load_files(train_filepaths)
    # 加载测试集 
    test_logvol_minus,test_logvol_plus = load_files(test_filepaths)
    

    X_train = np.array(train_logvol_minus, dtype=np.float32).reshape(-1,1)
    X_test = np.array(test_logvol_minus, dtype=np.float32).reshape(-1,1)

    # 转换目标变量为numpy数组
    y_train = np.array(train_logvol_plus, dtype=np.float32)
    y_test = np.array(test_logvol_plus, dtype=np.float32)

    print(f"训练集形状：{X_train.shape}，测试集形状：{X_test.shape}")

    return X_train, X_test, y_train, y_test

def model_train_LO(model,train_years, test_years):
    X_train, X_test, y_train, y_test=prepare_data_LO(train_years, test_years)
    model.fit(X_train,y_train)

    train_score=model.score(X_train,y_train)
    test_score=model.score(X_test,y_test)

    coeff_used = np.sum(model.coef_!=0)
    print ("training score : ", train_score )
    print ("test score     : ", test_score)
    print ("number of features used: ", coeff_used)

    mse, rmse, r2, pred=evaluate_model(model, X_test, y_test)
    return mse, rmse, r2, pred


#Lasso模型

lasso = Lasso(alpha=0.0001, max_iter=100000)

# TF-IDF+Lasso


print(f'  测试2001   ')
mse1, rmse1, r21, pred1=model_train(lasso,list(range(1996,2001)), [2001])

print(f'  测试2002   ')
mse2, rmse2, r22, pred2=model_train(lasso,list(range(1997,2002)), [2002])

print(f'  测试2003   ')
mse3, rmse3, r23, pred3=model_train(lasso,list(range(1998,2003)), [2003])

print(f'  测试2004   ')
mse4, rmse4, r24, pred4=model_train(lasso,list(range(1999,2004)), [2004])

print(f'  测试2005   ')
mse5, rmse5, r25, pred5=model_train(lasso,list(range(2000,2005)), [2005])

print(f'  测试2006   ')
mse6, rmse6, r26, pred6=model_train(lasso,list(range(2001,2006)), [2006])




# 打印结果
print('\n测试结果:')
print(f'  2001   - MSE: {mse1:.4f}, RMSE: {rmse1:.4f}, R²: {r21:.4f}')
print(f'  2002   - MSE: {mse2:.4f}, RMSE: {rmse2:.4f}, R²: {r22:.4f}')
print(f'  2003   - MSE: {mse3:.4f}, RMSE: {rmse3:.4f}, R²: {r23:.4f}')
print(f'  2004   - MSE: {mse4:.4f}, RMSE: {rmse4:.4f}, R²: {r24:.4f}')
print(f'  2005   - MSE: {mse5:.4f}, RMSE: {rmse5:.4f}, R²: {r25:.4f}')
print(f'  2006   - MSE: {mse6:.4f}, RMSE: {rmse6:.4f}, R²: {r26:.4f}')
