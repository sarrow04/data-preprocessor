# -*- coding: utf-8 -*-
"""preprocessing_app_v9_robust_date_clean

"""

import streamlit as st
import pandas as pd
import io
import plotly.express as px
import numpy as np
import mojimoji
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- Streamlitアプリの基本設定 ---
st.set_page_config(page_title="データ前処理サポーター", page_icon="🛠️", layout="wide")
st.title("🛠️ データ前処理サポーター")
st.write("CSVファイルをアップロードするだけで、データの健康診断とクリーニングができます。")

# --- Session Stateの初期化 ---
if 'df' not in st.session_state: st.session_state.df = None
if 'original_df' not in st.session_state: st.session_state.original_df = None
if 'target_col' not in st.session_state: st.session_state.target_col = None
if 'feature_cols' not in st.session_state: st.session_state.feature_cols = None


# --- サイドバー ---
with st.sidebar:
    st.header("1. ファイルをアップロード")
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=['csv'])
    if uploaded_file is not None:
        if "uploaded_file_name" not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.df = df
                st.session_state.original_df = df.copy()
                st.session_state.target_col = None
                st.session_state.feature_cols = None
                st.sidebar.success("ファイルが正常に読み込まれました！")
            except Exception as e: st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")

    if st.session_state.df is not None:
        df_sidebar = st.session_state.df
        st.header('2. 前処理ツール')

        st.subheader('列の四則演算')
        numeric_cols_sidebar = df_sidebar.select_dtypes(include=np.number).columns.tolist()
        operation = st.selectbox('実行したい操作を選択', ['---', '列の合計', '列の積', '列の差', '列の商'])
        if operation != '---':
            if operation in ['列の差', '列の商']:
                cols_to_operate = st.multiselect(f'「{operation}」を計算する数値列を2つ選択', numeric_cols_sidebar, max_selections=2)
            else:
                cols_to_operate = st.multiselect(f'「{operation}」を計算する数値列を2つ以上選択', numeric_cols_sidebar)
            new_col_name = st.text_input('新しい列の名前を入力してください', f'new_{operation}')
            if st.button(f'{operation}を実行'):
                if len(cols_to_operate) >= 2 and new_col_name:
                    try:
                        temp_df = st.session_state.df.copy()
                        if operation == '列の合計': temp_df[new_col_name] = temp_df[cols_to_operate].sum(axis=1)
                        elif operation == '列の積': temp_df[new_col_name] = temp_df[cols_to_operate].prod(axis=1)
                        elif operation == '列の差': temp_df[new_col_name] = temp_df[cols_to_operate[0]] - temp_df[cols_to_operate[1]]
                        elif operation == '列の商':
                            if (temp_df[cols_to_operate[1]] == 0).any(): st.error("エラー: 割る数に0が含まれています。")
                            else: temp_df[new_col_name] = temp_df[cols_to_operate[0]] / temp_df[cols_to_operate[1]]
                        st.session_state.df = temp_df
                        st.success(f"新しい列 '{new_col_name}' を作成しました。"); st.rerun()
                    except Exception as e: st.error(f"計算中にエラーが発生しました: {e}")
                else: st.warning("列と新しい列名を確認してください。")

        if st.button("最初の状態に戻す"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.target_col = None
            st.session_state.feature_cols = None
            st.info("データが最初の状態にリセットされました。")
            st.rerun()

# --- メイン画面 ---
if st.session_state.df is not None:
    df_main = st.session_state.df

    st.header("🩺 2. データの健康診断")
    tab1, tab2, tab3, tab4 = st.tabs(["基本情報", "欠損値", "統計量", "グラフで可視化"])
    with tab1:
        st.subheader("基本情報"); st.markdown(f"**行数:** {df_main.shape[0]} 行, **列数:** {df_main.shape[1]} 列")
        st.subheader("データプレビュー"); st.dataframe(df_main.head())
    with tab2:
        st.subheader("各列の欠損値の数"); missing_values = df_main.isnull().sum(); st.dataframe(missing_values[missing_values > 0].sort_values(ascending=False).rename("欠損数"))
    with tab3:
        st.subheader("各列の統計量"); st.dataframe(df_main.describe(include='all'))
    with tab4:
        st.subheader("列の分布をグラフで確認")
        graph_col = st.selectbox("グラフを表示する列を選択", df_main.columns, key="graph_col")
        if graph_col:
            if pd.api.types.is_numeric_dtype(df_main[graph_col]):
                st.write(f"**{graph_col}** のヒストグラム")
                fig = px.histogram(df_main, x=graph_col, title=f'「{graph_col}」の分布')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(f"**{graph_col}** の度数分布（上位20件）")
                value_counts = df_main[graph_col].value_counts().nlargest(20)
                value_counts_df = value_counts.reset_index()
                value_counts_df.columns = [graph_col, 'カウント']
                fig = px.bar(value_counts_df, x=graph_col, y='カウント', title=f'「{graph_col}」のTOP20カテゴリ')
                st.plotly_chart(fig, use_container_width=True)

    st.header("🧹 3. データ全体のクリーニング")
    st.subheader("列の一括削除")
    columns_to_drop = st.multiselect('不要な列を複数選択できます。', df_main.columns)
    if st.button("選択した列を削除する"):
        if columns_to_drop:
            st.session_state.df = df_main.drop(columns=columns_to_drop)
            st.success("選択された列を削除しました。"); st.rerun()
        else: st.warning("削除する列が選択されていません。")

    num_duplicates = df_main.duplicated().sum()
    if num_duplicates > 0:
        st.write(f"データセット全体に **{num_duplicates}** 件の重複行があります。")
        if st.button("重複行をすべて削除する"):
            st.session_state.df = df_main.drop_duplicates()
            st.success("重複行を削除しました。"); st.rerun()

    st.header("💊 4. 列ごとの対話型クリーニング")
    st.write("処理したい列を選択し、アクションを実行してください。")
    selected_column = st.selectbox("処理対象の列を選択してください", df_main.columns)
    if selected_column:
        col_type = df_main[selected_column].dtype; missing_count = df_main[selected_column].isnull().sum(); st.write(f"選択中の列: **{selected_column}** (データ型: {col_type}, 欠損値: {missing_count}個)")
        st.subheader(f"「{selected_column}」列へのアクション")
        if missing_count > 0:
            with st.expander("欠損値の処理"):
                fill_method = st.radio("欠損値をどうしますか？", ("平均値で埋める", "中央値で埋める", "最頻値で埋める", "指定した値で埋める", "行ごと削除する"), key=f"fill_{selected_column}")
                fill_value = None
                if fill_method == "指定した値で埋める": fill_value = st.text_input("埋める値を入力してください")
                if st.button("欠損値処理を実行", key=f"btn_fill_{selected_
