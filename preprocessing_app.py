# -*- coding: utf-8 -*-
"""
preprocessing_app_v22_final
IndentationErrorを修正した最終安定版
"""
import streamlit as st
import pandas as pd
import io
import plotly.express as px
import numpy as np
import mojimoji
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys

# --- 1. Streamlitアプリの基本設定 ---
st.set_page_config(page_title="データ前処理サポーター", page_icon="🛠️", layout="wide")

# --- 2. Session Stateの初期化 ---
if 'df' not in st.session_state: st.session_state.df = None
if 'original_df' not in st.session_state: st.session_state.original_df = None
if 'uploaded_file_name' not in st.session_state: st.session_state.uploaded_file_name = None
if 'target_col' not in st.session_state: st.session_state.target_col = None
if 'feature_cols' not in st.session_state: st.session_state.feature_cols = None

# --- 3. 各UIセクションの関数化 ---

def display_sidebar():
    """サイドバーのUIを表示し、ファイルアップロードや基本操作を処理する"""
    with st.sidebar:
        st.header("1. ファイルをアップロード")
        uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=['csv'])
        
        if uploaded_file is not None:
            if st.session_state.uploaded_file_name != uploaded_file.name:
                try:
                    df = pd.read_csv(uploaded_file, header=None)
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.df = df
                    st.session_state.original_df = df.copy()
                    st.session_state.target_col = None
                    st.session_state.feature_cols = None
                    st.sidebar.success("ファイルが正常に読み込まれました！")
                except Exception as e:
                    st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")

        if st.session_state.df is not None:
            st.header('2. 前処理ツール')
            with st.expander('列の四則演算'):
                df_sidebar = st.session_state.df
                numeric_cols_sidebar = [c for c in df_sidebar.columns if pd.api.types.is_numeric_dtype(df_sidebar[c])]
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
                st.info("データが最初の状態にリセットされました。"); st.rerun()

        st.subheader("🧪 環境情報")
        st.write(f"Pandas Version: **{pd.__version__}**")
        st.write(f"Python Version: {sys.version.split(' ')[0]}")


def display_health_check(df):
    """「データの健康診断」セクションを表示する"""
    st.header("🩺 データの健康診断")
    tab1, tab2, tab3, tab4 = st.tabs(["基本情報", "欠損値", "統計量", "グラフで可視化"])
    with tab1:
        st.subheader("基本情報"); st.markdown(f"**行数:** {df.shape[0]} 行, **列数:** {df.shape[1]} 列")
        st.subheader("データプレビュー"); st.dataframe(df.head())
    with tab2:
        st.subheader("各列の欠損値の数"); missing_values = df.isnull().sum(); st.dataframe(missing_values[missing_values > 0].sort_values(ascending=False).rename("欠損数"))
    with tab3:
        st.subheader("各列の統計量"); st.dataframe(df.describe(include='all'))
    with tab4:
        st.subheader("列の分布をグラフで確認")
        graph_col = st.selectbox("グラフを表示する列を選択", df.columns, key="graph_col")
        if graph_col is not None:
            plot_series = df[graph_col].dropna()
            if pd.api.types.is_numeric_dtype(plot_series) and not plot_series.empty:
                st.write(f"**{graph_col}** のヒストグラム")
                fig = px.histogram(df, x=graph_col, title=f'「{graph_col}」の分布')
                st.plotly_chart(fig, use_container_width=True)
            elif not plot_series.empty:
                st.write(f"**{graph_col}** の度数分布（上位20件）")
                value_counts = plot_series.value_counts().nlargest(20)
                value_counts_df = value_counts.reset_index()
                value_counts_df.columns = [str(graph_col), 'カウント']
                fig = px.bar(value_counts_df, x=str(graph_col), y='カウント', title=f'「{graph_col}」のTOP20カテゴリ')
                st.plotly_chart(fig, use_container_width=True)

def display_global_cleaning(df):
    """「データ全体のクリーニング」セクションを表示する"""
    st.header("🧹 データ全体のクリーニング")
    st.subheader("先頭行の削除（ヘッダー行の指定）")
    st.write("CSVファイルの上部に説明書きなどが含まれている場合、データ本体が始まる行を指定して不要な行を削除します。")
    
    header_row = st.number_input(
        "新しいヘッダー（列名）として使用したい行の番号を入力してください（0から始まります）",
        min_value=0, max_value=len(df)-2 if len(df) > 1 else 0, value=0, step=1,
        help="例えば「4」と入力すると、0〜3行目が削除され、4行目が新しい列名になります。"
    )

    if st.button("指定行をヘッダーとして設定し、それより上を削除"):
        if header_row > 0:
            try:
                df_copy = df.copy()
                new_header = df_copy.iloc[header_row]
                df_copy = df_copy.iloc[header_row+1:]
                df_copy.columns = new_header
                df_copy.reset_index(drop=True, inplace=True)
                st.session_state.df = df_copy
                st.success(f"{header_row}行目を新しいヘッダーに設定し、それより上の行を削除しました。")
                st.rerun()
            except Exception as e: st.error(f"処理中にエラーが発生しました: {e}")
        else: st.info("0行目が選択されているため、処理は実行されませんでした。")
    st.markdown("---")

    st.subheader("列の一括削除")
    columns_to_drop = st.multiselect('不要な列を複数選択できます。', df.columns)
    if st.button("選択した列を削除する"):
        if columns_to_drop:
            st
