# -*- coding: utf-8 -*-
"""
preprocessing_app_v25_final
列ごとの処理において、最初のデータ行を除外するオプションを追加した最終版
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
                df = None
                try:
                    df = pd.read_csv(uploaded_file, header=None)
                except UnicodeDecodeError:
                    try:
                        st.sidebar.warning("UTF-8での読み込みに失敗。Shift-JISで再試行します。")
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, header=None, encoding='cp932')
                    except Exception as e:
                        st.error(f"Shift-JISでも読み込みに失敗しました: {e}")
                except Exception as e:
                    st.error(f"ファイルの読み込み中に予期せぬエラーが発生しました: {e}")
                
                if df is not None:
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.df = df
                    st.session_state.original_df = df.copy()
                    st.session_state.target_col = None
                    st.session_state.feature_cols = None
                    st.sidebar.success("ファイルが正常に読み込まれました！")
                else:
                    st.session_state.df = None

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
        try:
            df_copy = df.copy()
            new_header = df_copy.iloc[header_row].astype(str)
            df_copy = df_copy.iloc[header_row+1:]
            df_copy.columns = new_header
            df_copy.reset_index(drop=True, inplace=True)
            st.session_state.df = df_copy
            st.success(f"{header_row}行目を新しいヘッダーに設定し、データフレームを更新しました。")
            st.rerun()
        except Exception as e:
            st.error(f"処理中にエラーが発生しました: {e}")
    st.markdown("---")

    st.subheader("列の一括削除")
    columns_to_drop = st.multiselect('不要な列を複数選択できます。', df.columns)
    if st.button("選択した列を削除する"):
        if columns_to_drop:
            st.session_state.df = df.drop(columns=columns_to_drop)
            st.success("選択された列を削除しました。"); st.rerun()
        else: st.warning("削除する列が選択されていません。")

    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        st.subheader("重複行の削除")
        st.write(f"データセット全体に **{num_duplicates}** 件の重複行があります。")
        if st.button("重複行をすべて削除する"):
            st.session_state.df = df.drop_duplicates(); st.success("重複行を削除しました。"); st.rerun()

def display_column_wise_cleaning(df):
    st.header("💊 列ごとの対話型クリーニング")
    st.write("処理したい列を選択し、アクションを実行してください。")
    selected_column = st.selectbox("処理対象の列を選択してください", df.columns)
    
    if selected_column is None: return

    # ▼▼▼ 新機能: 最初の行を除外するチェックボックス ▼▼▼
    exclude_first_row = st.checkbox("最初のデータ行（0行目）を処理から除外する", key=f"exclude_first_{selected_column}", help="ヘッダー直下の単位などが記載された行を、変換や計算などの処理対象から外します。")
    st.markdown("---")
    # ▲▲▲ ここまで ▲▲▲

    col_type = df[selected_column].dtype
    missing_count = df[selected_column].isnull().sum()
    st.write(f"選択中の列: **{selected_column}** (データ型: {col_type}, 欠損値: {missing_count}個)")
    st.subheader(f"「{selected_column}」列へのアクション")

    if missing_count > 0:
        with st.expander("欠損値の処理"):
            options = ["最頻値で埋める", "指定した値で埋める", "行ごと削除する"]
            if pd.api.types.is_numeric_dtype(df[selected_column]):
                options = ["平均値で埋める", "中央値で埋める"] + options
            fill_method = st.radio("欠損値をどうしますか？", options, key=f"fill_{selected_column}")
            fill_value = None
            if fill_method == "指定した値で埋める": fill_value = st.text_input("埋める値を入力してください")
            if st.button("欠損値処理を実行", key=f"btn_fill_{selected_column}"):
                df_copy = df.copy()
                series_to_process = df_copy[selected_column].iloc[1:] if exclude_first_row else df_copy[selected_column]
                
                if fill_method == "平均値で埋める": series_to_process.fillna(series_to_process.mean(), inplace=True)
                elif fill_method == "中央値で埋める": series_to_process.fillna(series_to_process.median(), inplace=True)
                elif fill_method == "最頻値で埋める": series_to_process.fillna(series_to_process.mode()[0], inplace=True)
                elif fill_method == "指定した値で埋める" and fill_value: series_to_process.fillna(fill_value, inplace=True)
                elif fill_method == "行ごと削除する": series_to_process.dropna(inplace=True) # Note: this will only affect the slice
                
                df_copy[selected_column].update(series_to_process)
                if fill_method == "行ごと削除する": # For dropna, we need to handle the whole dataframe
                    subset_to_check = df_copy.index.isin(series_to_process.index)
                    if exclude_first_row:
                        indices_to_drop = df_copy.index[1:][~subset_to_check[1:]]
                    else:
                        indices_to_drop = df_copy.index[~subset_to_check]
                    df_copy.drop(indices_to_drop, inplace=True)

                st.session_state.df = df_copy
                st.success(f"「{selected_column}」列の欠損値処理が完了しました。"); st.rerun()

    with st.expander("データ型の変換（数値・文字列）"):
        new_type = st.selectbox("変換したいデータ型を選択", ["---", "数値 (int)", "数値 (float)", "文字列 (str)"], key=f"type_{selected_column}")
        if st.button("データ型を変換", key=f"btn_type_{selected_column}"):
            if new_type != "---":
                try:
                    df_copy = df.copy()
                    series_to_modify = df_copy[selected_column]
                    target_slice = series_to_modify.iloc[1:] if exclude_first_row and len(series_to_modify) > 0 else series_to_modify
                    pre_missing = target_slice.isnull().sum()

                    if new_type in ["数値 (int)", "数値 (float)"]:
                        processed_slice = pd.to_numeric(target_slice.astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
                        if new_type == "数値 (int)": processed_slice = processed_slice.astype('Int64')
                    elif new_type == "文字列 (str)":
                        processed_slice = target_slice.astype(str)
                    
                    final_series = series_to_modify.copy()
                    final_series.update(processed_slice)
                    df_copy[selected_column] = final_series
                    
                    post_missing = final_series.isnull().sum()
                    st.session_state.df = df_copy
                    st.success(f"「{selected_column}」列を{new_type}型に変換しました。")
                    if post_missing > pre_missing: st.warning(f"{post_missing - pre_missing}個のデータが変換に失敗し、欠損値になりました。")
                    st.rerun()
                except Exception as e: st.error(f"変換に失敗しました: {e}")
    
    with st.expander("日付型への変換"):
        date_format_option = st.radio("データの形式を選択", ("標準的な形式 (例: 2023-01-01, 2023/1/1)", "日本の形式 (例: 2023年1月1日, 令和5年1月1日)", "区切り文字なし (例: 20230101)", "Excelのシリアル値 (例: 45123)"), key=f"date_{selected_column}")
        if st.button("日付型に変換を実行", key=f"btn_date_{selected_column}"):
            try:
                df_copy = df.copy()
                col_name = selected_column
                series_to_modify = df_copy[col_name]
                target_slice = series_to_modify.iloc[1:] if exclude_first_row and len(series_to_modify) > 0 else series_to_modify
                pre_missing = target_slice.isnull().sum()
                
                converted_slice = None
                if date_format_option == "Excelのシリアル値 (例: 45123)":
                    numeric_series = pd.to_numeric(target_slice, errors='coerce')
                    converted_slice = pd.to_datetime(numeric_series, unit='D', origin='1899-12-30')
                else:
                    s = target_slice.astype(str).dropna()
                    s = s.apply(lambda x: mojimoji.zen_to_han(x, kana=False))
                    s = s.str.replace(r'\s+', '', regex=True)
                    if date_format_option == "標準的な形式 (例: 2023-01-01, 2023/1/1)":
                        res1 = pd.to_datetime(s, errors='coerce')
                        res2 = pd.to_datetime(s, format='%Y-%m', errors='coerce')
                        res3 = pd.to_datetime(s, format='%Y/%m', errors='coerce')
                        converted_slice = res1.fillna(res2).fillna(res3)
                    elif date_format_option == "日本の形式 (例: 2023年1月1日, 令和5年1月1日)":
                        def convert_japanese_date(jp_date_text):
                            if not isinstance(jp_date_text, str): return None
                            text = jp_date_text.replace('元年', '1年')
                            try: return pd.to_datetime(text, format='%Y年%m月%d日')
                            except ValueError:
                                try: return pd.to_datetime(text, format='%Y年%m月')
                                except ValueError:
                                    year_str = text.split('年')[0]; year = 0
                                    if '令和' in year_str: year = int(year_str.replace('令和', '')) + 2018
                                    elif '平成' in year_str: year = int(year_str.replace('平成', '')) + 1988
                                    elif '昭和' in year_str: year = int(year_str.replace('昭和', '')) + 1925
                                    elif '大正' in year_str: year = int(year_str.replace('大正', '')) + 1911
                                    elif '明治' in year_str: year = int(year_str.replace('明治', '')) + 1867
                                    if year == 0: return None
                                    month_day_part = text.split('年')[1]
                                    if '日' in month_day_part:
                                        month = int(month_day_part.split('月')[0])
                                        day = int(month_day_part.split('月')[1].replace('日', ''))
                                    else:
                                        month = int(month_day_part.replace('月', '')); day = 1
                                    return pd.to_datetime(f'{year}-{month}-{day}')
                        converted_slice = s.apply(convert_japanese_date)
                    elif date_format_option == "区切り文字なし (例: 20230101)":
                        converted_slice = pd.to_datetime(s, format='%Y%m%d', errors='coerce')
                
                final_series = series_to_modify.copy()
                if converted_slice is not None:
                    final_series.update(converted_slice)
                
                df_copy[col_name] = final_series
                post_missing = df_copy[col_name].isnull().sum()
                st.session_state.df = df_copy; st.success("日付型への変換が完了しました。")
                if post_missing > pre_missing: st.warning(f"{post_missing - pre_missing}個のデータが変換に失敗し、欠損値になりました。")
                st.rerun()
            except Exception as e: st.error(f"変換中にエラーが発生しました: {e}")

    if pd.api.types.is_string_dtype(df[selected_column]):
        with st.expander("文字列のクレンジング"):
            clean_option = st.selectbox("実行したいクレンジングを選択", ["---", "前後の空白を削除", "すべて小文字に変換", "すべて大文字に変換", "全角英数記号を半角に変換"], key=f"clean_{selected_column}")
            if st.button("文字列クレンジングを実行", key=f"btn_clean_{selected_column}"):
                if clean_option != "---":
                    df_copy = df.copy()
                    series_to_modify = df_copy[selected_column]
                    target_slice = series_to_modify.iloc[1:] if exclude_first_row and len(series_to_modify) > 0 else series_to_modify
                    
                    col = target_slice.astype(str)
                    if clean_option == "前後の空白を削除": processed_slice = col.str.strip()
                    elif clean_option == "すべて小文字に変換": processed_slice = col.str.lower()
                    elif clean_option == "すべて大文字に変換": processed_slice = col.str.upper()
                    elif clean_option == "全角英数記号を半角に変換": processed_slice = col.apply(lambda x: mojimoji.zen_to_han(x, kana=False))
                    
                    final_series = series_to_modify.copy()
                    final_series.update(processed_slice)
                    df_copy[selected_column] = final_series
                    
                    st.session_state.df = df_copy; st.success(f"「{selected_column}」列の「{clean_option}」を実行しました。"); st.rerun()

def display_feature_engineering(df):
    st.header("🧮 特徴量エンジニアリング")
    st.write("機械学習モデルで使いやすいようにデータを変換します。")
    with st.expander("ワンホットエンコーディング"):
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        ohe_cols = st.multiselect("ワンホットエンコーディングを適用したい列を複数選択", categorical_cols, key="ohe_cols")
        if st.button("ワンホットエンコーディングを実行"):
            if ohe_cols:
                st.session_state.df = pd.get_dummies(df, columns=ohe_cols, dtype=float)
                st.success("ワンホットエンコーディングを実行しました。"); st.rerun()
            else: st.warning("列が選択されていません。")
    with st.expander("正規化・標準化"):
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        scaling_method = st.radio("手法を選択してください", ("最小最大正規化 (Min-Max Scaling)", "標準化 (Standardization)"), key="scaling_method")
        numeric_cols_selected = st.multiselect("適用したい数値列を複数選択", numeric_cols, key="scaling_cols")
        if st.button("正規化・標準化を実行"):
            if numeric_cols_selected:
                df_copy = df.copy()
                if scaling_method == "最小最大正規化 (Min-Max Scaling)": scaler = MinMaxScaler()
                else: scaler = StandardScaler()
                df_copy[numeric_cols_selected] = scaler.fit_transform(df_copy[numeric_cols_selected])
                st.session_state.df = df_copy
                st.success(f"「{scaling_method}」を実行しました。"); st.rerun()
            else: st.warning("列が選択されていません。")

def display_variable_settings(df):
    st.header("🎯 目的変数と説明変数の設定")
    st.write("モデル学習に使用する変数（列）の役割を定義します。")
    if st.session_state.target_col:
        st.info(f"現在の設定 - 目的変数: **{st.session_state.target_col}**, 説明変数: **{len(st.session_state.feature_cols)}** 個")
    all_columns = df.columns.tolist()
    target_options = ["---"] + all_columns
    selected_target = st.selectbox("予測したい「目的変数」を1つ選択してください", target_options, index=0, key="target_select")
    if selected_target != "---":
        available_features = [col for col in all_columns if col != selected_target]
        selected_features = st.multiselect("予測に使う「説明変数」を1つ以上選択してください", available_features, default=available_features, key="feature_select")
    else:
        st.multiselect("予測に使う「説明変数」を1つ以上選択してください", ["まず目的変数を選択してください"], disabled=True, key="feature_select_disabled")
    if st.button("変数の役割を設定"):
        if selected_target != "---" and len(selected_features) > 0:
            st.session_state.target_col = selected_target
            st.session_state.feature_cols = selected_features
            st.success("目的変数と説明変数を設定しました。"); st.rerun()
        else: st.warning("目的変数と説明変数を正しく選択してください。")

def display_download_button(df):
    st.header("✅ 処理済みデータのダウンロード")
    @st.cache_data
    def convert_df_to_csv(df_to_convert):
        return df_to_convert.to_csv(index=False).encode('utf-8-sig')
    csv = convert_df_to_csv(df)
    st.download_button(label="整形済みデータをCSVでダウンロード", data=csv, file_name='cleaned_data.csv', mime='text/csv')

def main():
    st.title("🛠️ データ前処理サポーター")
    st.write("CSVファイルをアップロードするだけで、データの健康診断とクリーニングができます。")
    display_sidebar()
    if st.session_state.df is not None:
        df_main = st.session_state.df
        display_health_check(df_main)
        display_global_cleaning(df_main)
        display_column_wise_cleaning(df_main)
        display_feature_engineering(df_main)
        display_variable_settings(df_main)
        display_download_button(df_main)
    else:
        st.info("サイドバーからCSVファイルをアップロードして分析を開始してください。")

if __name__ == "__main__":
    main()
