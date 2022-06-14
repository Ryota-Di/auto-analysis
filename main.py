import streamlit as streamlit
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split


st.markdown("# 自動分析ツール")

st.caption("csvをアップロードするだけで、基本的なビジュアライズから機械学習を用いた予測までを簡単に行うことができます。")

st.markdown("## csvの読み込み")
if st.checkbox("csvを追加"):
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        global df
        # bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)
        df = pd.read_csv(uploaded_file)

    # EDA開始
    st.markdown("## EDA")
    # headの表示
    if st.checkbox("head"):
        st.write(df.head(3))

    # shapeの表示
    if st.checkbox("shape"):
        st.write(df.shape)

    # infoの表示
    # できず

    # lineplotの表示
    if st.checkbox("lineplot"):
        options = st.multiselect("折れ線グラフで表すカラムの追加",df.columns)
        if len(options)>0:
            view = df[options]
            st.line_chart(view)

    # pairplotの表示
    if st.checkbox("pairplot"):
        options = st.multiselect("相関図で表すカラムの追加",df.columns)
        # print(options)
        if len(options)>1:
            fig = sns.pairplot(df,vars = options)
            st.pyplot(fig)

    # 相関係数の表示
    if st.checkbox("corr"):
        st.write(df.corr())

    st.markdown("## 前処理")

    # 自動変換後の表示 
    # 自動変換とは文字列データを削除し、カテゴリーデータを数値データに変換すること

    if st.checkbox("自動変換後表示"):
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
        option = st.radio("寄与対象",df_toInt.columns)
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
                'metric': {'rmse'}, # 評価指標 : rsme(平均二乗誤差の平方根) 
        }

        # モデルの学習
        model = lgb.train(params,
                        train_set=lgb_train, # トレーニングデータの指定
                        valid_sets=lgb_eval, # 検証データの指定
                        )

        importance = pd.DataFrame(model.feature_importance(), index=X.columns, columns=['importance'])
        st.write(importance)