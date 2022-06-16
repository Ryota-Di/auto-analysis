import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split


st.markdown("# 自動分析ツール")

st.caption("csvをアップロードするだけで、Excelより手軽に可視化")
st.caption("表やグラフは随時簡単に保存できます！")

st.markdown("### csvの読み込み")
uploaded_file = st.file_uploader("ファイルを選択してください",type="csv")
df = pd.DataFrame()
if uploaded_file is not None:
    # global df
    # bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)
    df = pd.read_csv(uploaded_file)
st.caption("別ファイルがアップロードされると上書きされてしまいますのでご注意ください")

# 可視化開始
st.markdown("### データの可視化")

# headの表示
if st.checkbox("最初の5行を表示"):
    if uploaded_file is None:
        st.error("データがアップロードされていません")
    else:
        st.write(df.head(5))

# shapeの表示
if st.checkbox("データの行数・列数を表示"):
    if uploaded_file is None:
        st.error("データがアップロードされていません")
    else:
        st.write(df.shape)

if st.checkbox("基本統計量を表示"):
    if uploaded_file is None:
        st.error("データがアップロードされていません")
    else:
        st.write(df.describe())

# infoの表示
# できず

# lineplotの表示
if st.checkbox("折れ線グラフ"):
    if uploaded_file is None:
        st.error("データがアップロードされていません")
    else:
        options = st.multiselect("折れ線グラフで表すカラムの追加",df.columns)
        if len(options)>0:
            view = df[options]
            st.line_chart(view)

# pairplotの表示
if st.checkbox("散布図行列"):
    if uploaded_file is None:
        st.error("データがアップロードされていません")
    else:
        options = st.multiselect("散布図行列で表すカラムの追加",df.columns)
        # print(options)
        if len(options)>1:
            fig = sns.pairplot(df,vars = options)
            st.pyplot(fig)

# 相関係数の表示
if st.checkbox("相関係数"):
    if uploaded_file is None:
        st.error("データがアップロードされていません")
    else:
        # st.write(df.corr())
        st.write(df.corr().style.background_gradient(cmap='Reds'))

st.markdown("### 前処理")

# 自動変換後の表示 
# 自動変換とは文字列データを削除し、カテゴリーデータを数値データに変換すること

if st.checkbox("自動変換後表示"):
    if uploaded_file is None:
        st.error("データがアップロードされていません")
    else:
        global df_toInt 
        df_toInt = df
        object_columns = df.select_dtypes(include=object).columns
        for i in object_columns:
            if df[i].nunique()<5:
                df_toInt = pd.get_dummies(df_toInt, columns=[i], drop_first=True)
            else:
                df_toInt = df_toInt.drop(i,axis=1)
        df_toInt = df_toInt.dropna()
        st.write(df_toInt)

# 寄与度把握
if st.checkbox("寄与度把握"):
    if uploaded_file is None:
        st.error("データがアップロードされていません")
    else:
        option = st.radio("寄与対象",df_toInt.columns)
        if option is not None:
            y = df_toInt[option]
            X = df_toInt.drop(option,axis=1)

            # トレーニングデータ,テストデータの分割
            X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=2)


            # 学習に使用するデータを設定
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train) 

            # LightGBM parameters
            params = {
                    'task': 'train',
                    'boosting_type': 'gbdt',
                    'objective': 'binary', # 目的 : 二値分類
                    'metric': {'binary_logloss'}, # 評価指標 : logloss
            }

            # モデルの学習
            model = lgb.train(params,
                            train_set=lgb_train, # トレーニングデータの指定
                            valid_sets=lgb_eval, # 検証データの指定
                            early_stopping_rounds=10
                            )

            importance = pd.DataFrame(model.feature_importance(), index=X.columns, columns=['importance'])
            st.write(importance)