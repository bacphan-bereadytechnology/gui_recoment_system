import streamlit as st
import pandas as pd
import pickle
from surprise import *
import re
from pyvi import ViTokenizer
from gensim import *

DEFAULT_IMAGE_URL = "image4-1.jpeg"

# sidebar menu

# Khởi tạo menu mặc định
menu = ["Home", "About Project", "Collaborative Filtering", "Content Based Filtering"]

# Sử dụng session_state để điều hướng
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar menu dạng list
sidebar_choice = st.sidebar.radio("📂 Menu", menu, index=menu.index(st.session_state.page))

# Cập nhật khi chọn từ menu
if sidebar_choice != st.session_state.page:
    st.session_state.page = sidebar_choice
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ **Thực hiện bởi:**")
st.sidebar.markdown("- Phan Ngọc Phương Bắc\n- Nguyễn Tuấn Anh")
st.sidebar.markdown("### 👩‍🏫 **Giảng viên:**\n Cô Khuất Thùy Phương")
st.sidebar.markdown(
    """
    <hr>
    <div style='font-size: 12px; color: gray;'>
        Source Code: <a href="https://github.com/bacphan-bereadytechnology/gui_recoment_system" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

if st.session_state.page == "Home":

    st.title("🏠 Recommendation System")
    st.markdown("Chọn một trong các chức năng dưới đây:")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 About Project"):
            st.session_state.page = "About Project"
            # st.session_state.navigated_from_home = True
            st.rerun()
    with col2:
        if st.button("🛍️ Collaborative Filtering"):
            st.session_state.page = "Collaborative Filtering"
            # st.session_state.navigated_from_home = True
            st.rerun()
    with col3:
        if st.button("🔍 Content Based Filtering"):
            st.session_state.page = "Content Based Filtering"
            # st.session_state.navigated_from_home = True
            st.rerun()


elif st.session_state.page == "About Project":

    st.title("Mục tiêu của dự án")
    st.markdown("""
        ### 🎯 Mục tiêu
        Xây dựng hệ thống gợi ý sản phẩm thông minh nhằm nâng cao trải nghiệm người dùng và tăng khả năng mua hàng trên nền tảng thương mại điện tử.

        ---

        ### Công nghệ sử dụng

        #### 🛍️ 1. SVD – Collaborative Filtering
        - Áp dụng mô hình SVD từ thư viện `Surprise`.
        - Dựa vào hành vi đánh giá sản phẩm của người dùng để gợi ý cá nhân hoá.

        #### 🔍 2. Gensim – Content-Based Filtering
        - Vector hóa nội dung mô tả sản phẩm bằng **TF-IDF** (`gensim.models.TfidfModel`).
        - Tính toán độ tương đồng giữa sản phẩm bằng **cosine similarity** qua `SparseMatrixSimilarity`.

        ---

        ### 🧪 Tính năng chính

        - 🔍 Tìm kiếm sản phẩm theo nội dung người dùng nhập.
        - 📦 Gợi ý sản phẩm tương tự dựa trên sản phẩm đã chọn.
        - 🙋 Gợi ý sản phẩm cá nhân hoá theo người dùng (`user_id`).
        - ⚙️ Giao diện trực quan và dễ dùng với **Streamlit**.
        """)

    

elif st.session_state.page == "Collaborative Filtering":

    def recommend_products_by_subcategory(
        rating_df,
        products_df,
        model,
        userId,
        selected_sub_category,
        user_rating=3,
        estimateScore=3,
        num_recommendations=10
    ):
        # Lọc sản phẩm theo sub_category đã chọn
        filtered_products = products_df[products_df["sub_category"] == selected_sub_category]
        available_product_ids = set(filtered_products["product_id"].unique())

        # Lọc sản phẩm đã đánh giá
        rated_product_ids = rating_df[
            (rating_df["user_id"] == userId) & (rating_df["rating"] >= user_rating)
        ]["product_id"].unique()

        # Danh sách sản phẩm chưa đánh giá và thuộc sub_category đã chọn
        unrated_product_ids = [
            pid for pid in available_product_ids if pid not in rated_product_ids
        ]

        # Dự đoán điểm
        predictions = []
        for pid in unrated_product_ids:
            try:
                est = model.predict(userId, pid).est
                if est >= estimateScore:
                    predictions.append((pid, est))
            except Exception:
                continue

        # Kết quả dự đoán
        df_score = pd.DataFrame(predictions, columns=["product_id", "EstimateScore"])
        df_score = df_score.sort_values(by="EstimateScore", ascending=False).head(num_recommendations)

        # Gộp thêm thông tin sản phẩm
        result_df = df_score.merge(products_df, on="product_id", how="left")

        return result_df

    @st.cache_data
    def load_ratings():
        return pd.read_csv("rating.csv")

    @st.cache_data
    def load_products():
        return pd.read_csv("product_clean.csv")
    
    @st.cache_data
    def load_subcategory():
        return pd.read_csv("sub_category_list.csv")

    @st.cache_resource
    def load_model():
        with open("svd_model.pkl", "rb") as file:
            return pickle.load(file)
        
    st.title("🛍️ Collaborative Filtering Recommendation System")
    st.write("Chọn người dùng để nhận gợi ý cá nhân hóa.")

    # Load dữ liệu
    ratings_df = load_ratings()
    products_df = load_products()
    subcategory_df = load_subcategory()

    # Load model
    model = load_model()

    top_n = st.slider("Số sản phẩm đề xuất:", 1, 20, 6, 1)
    min_rating = st.slider("Rating tối thiểu đã đánh giá:", 1, 5, 3, 1)
    min_estimate = st.slider("Điểm dự đoán tối thiểu:", 1, 5, 3, 1)

    df_KH = ratings_df[['user_id', 'user']].drop_duplicates()

    # Nếu chưa có danh sách hoặc người dùng nhấn nút thì random lại
    if 'df_sample' not in st.session_state or st.session_state.get('refresh_users', False):
        st.session_state.df_sample = df_KH.sample(5)
        st.session_state.refresh_users = False  # reset lại trigger

    df_sample = st.session_state.df_sample.reset_index(drop=True)
    user_display = df_sample.set_index('user_id')['user'].to_dict()

    st.write(df_sample)
    # Nút để random lại danh sách user
    if st.button("🔁 Random danh sách khách hàng khác"):
        st.session_state.refresh_users = True
        st.rerun()
    st.markdown("-"*20)
    col1, col2 = st.columns(2)

    with col1:
        # Selectbox với danh sách đã lưu
        selected_user = st.selectbox(
            "Chọn khách hàng:",
            options=list(user_display.keys()),
            format_func=lambda x: user_display[x]
        )

        st.write("🧑 Khách hàng đã chọn:", user_display[selected_user])
        st.write("🔑 Mã user_id:", selected_user)
    with col2:
        selected_category = st.selectbox(
            "Chọn danh mục sản phẩm:",
            options= subcategory_df["sub_category"].dropna().unique().tolist()
        )
        st.write("📁 Danh mục đã chọn:", selected_category)

    st.markdown("-"*20)

    # Gợi ý sản phẩm
    recommended_df = recommend_products_by_subcategory(
        rating_df=ratings_df,
        products_df=products_df,
        model=model,
        userId=selected_user,
        selected_sub_category=selected_category,
        user_rating=min_rating,
        estimateScore=min_estimate,
        num_recommendations=top_n,
    )

    # Gộp với dữ liệu sản phẩm
    # result_df = recommended_df.merge(products_df, on="product_id", how="left")

    st.subheader("🔮 Sản phẩm được đề xuất:")
    # Nếu không có sản phẩm được đề xuất
    if recommended_df.empty:
        st.info("Không có sản phẩm nào phù hợp với điều kiện lọc.")
    else:
        # Chia layout theo hàng ngang (3 cột mỗi hàng)
        cols = st.columns(3)

        for idx, row in recommended_df.iterrows():
            col = cols[idx % 3]  # Chia đều vào 3 cột

            with col:
                img_url = row.get("image", "")
                if isinstance(img_url, str) and img_url.startswith("http"):
                    st.image(img_url, use_container_width=True)
                else:
                    st.image(DEFAULT_IMAGE_URL, use_container_width=True)
                    
                st.markdown(f"**{row['product_name']}**")
                st.markdown(f"💰 **Giá:** {row['price']:,} đ")
                st.markdown(f"⭐ **Dự đoán:** {row['EstimateScore']:.2f}")
                st.markdown(f"📄 *{row['description'][:100]}...*")
                st.markdown("---")

elif st.session_state.page == "Content Based Filtering":

    @st.cache_resource
    def load_model_and_data():
        with open("gensim_bundle.pkl", "rb") as f:
            model_bundle = pickle.load(f)

        with open("data_bundle.pkl", "rb") as f:
            data_bundle = pickle.load(f)

        return model_bundle, data_bundle

    st.title("🔍 Content Based Filtering Recommendation System")
    st.write("Chọn sản phẩm để nhận gợi ý tương tự.")

    model_bundle, data_bundle = load_model_and_data()
    dictionary = model_bundle["dictionary"]
    tfidf = model_bundle["tfidf"]
    index = model_bundle["index"]
    df = data_bundle["df"]
    content_gem_re = data_bundle["content_gem_re"]

    # ======= Load stopwords =======
    with open("vietnamese-stopwords.txt", 'r', encoding='utf-8') as file:
        stop_words = file.read().split('\n')

    # ======= Hàm xử lý text =======
    def pprocess_text(text, stopwords=[]):
        text = text.lower()
        tokens = ViTokenizer.tokenize(text).split()
        tokens = [re.sub('[0-9]+', '', token) for token in tokens]
        tokens = [token for token in tokens if token not in stop_words]
        return tokens

    def get_similar_products(sim, top_n=3, self_index=0):
        sorted_sim = sorted(enumerate(sim), key=lambda item: item[1], reverse=True)
        sorted_sim = [item for item in sorted_sim if item[0] != self_index]
        similar_indices = [i for i, _ in sorted_sim[:top_n]]
        return similar_indices

    def get_gensim_recommendations(product_id, top_n=3):
        product_index = df.index[df['product_id'] == product_id][0]
        view_content = content_gem_re[product_index]
        kw_vector = dictionary.doc2bow(view_content)
        sim = index[tfidf[kw_vector]]
        similar_indices = get_similar_products(sim, top_n=top_n, self_index=product_index)
        return df.iloc[similar_indices]

    def text_to_similar_products(input_text, top_n=3):
        input_tokens = pprocess_text(input_text)
        kw_vector = dictionary.doc2bow(input_tokens)
        sim = index[tfidf[kw_vector]]
        sorted_sim = sorted(enumerate(sim), key=lambda item: item[1], reverse=True)
        similar_indices = [i for i, _ in sorted_sim[:top_n]]
        return df.iloc[similar_indices]
    
    # ======= Giao diện Streamlit =======
    tab1, tab2 = st.tabs(["📦 Gợi ý từ sản phẩm", "✍️ Gợi ý từ văn bản"])

    with tab1:
        
        top_n = st.slider("Số lượng sản phẩm tương tự", 1, 10, 6)

        df_product = df[['product_id', 'product_name']].drop_duplicates()

        # Nếu chưa có danh sách hoặc người dùng nhấn nút thì random lại
        if 'df_product' not in st.session_state or st.session_state.get('refresh_users', False):
            st.session_state.df_product = df_product.sample(5)
            st.session_state.refresh_users = False  # reset lại trigger

        df_product = st.session_state.df_product.reset_index(drop=True)
        product_display = df_product.set_index('product_id')['product_name'].to_dict()

        st.write(df_product)
        # Nút để random lại danh sách user
        if st.button("🔁 Random danh sách sản phẩm khác"):
            st.session_state.refresh_users = True
            st.rerun()

        st.markdown("-"*20)

        # Selectbox với danh sách đã lưu
        selected_product = st.selectbox(
            "Chọn sản phẩm:",
            options=list(product_display.keys()),
            format_func=lambda x: product_display[x]
        )

        st.write("📦 sản phẩm đã chọn:", product_display[selected_product])
        st.write("🔑 Mã product_id:", selected_product)

        result = get_gensim_recommendations(selected_product, top_n=top_n)

        st.subheader("🛍️ Kết quả sản phẩm:")
        st.markdown("-"*20)
        if result.empty:
            st.info("Không có sản phẩm nào phù hợp với điều kiện lọc.")
        else:
            for i in range(0, len(result), 3):  # Duyệt từng nhóm 3 sản phẩm
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(result):
                        row = result.iloc[i + j]
                        col = cols[j]

                        with col:
                            img_url = row.get("image", "")
                            if isinstance(img_url, str) and img_url.startswith("http"):
                                st.image(img_url, use_container_width=True)
                            else:
                                st.image(DEFAULT_IMAGE_URL, use_container_width=True)

                            st.markdown(f"**{row['product_name']}**")
                            st.markdown(f"💰 **Giá:** {row['price']:,} đ")
                            st.markdown(f"📄 *{row['description'][:100]}...*")
                            st.markdown("---")

    with tab2:

        top_n = st.slider("Số lượng gợi ý", 1, 10, 5, key="text_input")

        input_text = st.text_input("Nhập nội dung tìm kiếm:")
      
        
        if input_text.strip() == "":
            st.warning("Vui lòng nhập từ khóa.")
        else:
            result = text_to_similar_products(input_text, top_n=top_n)
            
        st.subheader("📌 Kết quả:")
        st.markdown("-"*20)
        if result.empty:
            st.info("Không có sản phẩm nào phù hợp với điều kiện lọc.")
        else:
            for i in range(0, len(result), 3):  # Duyệt từng nhóm 3 sản phẩm
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(result):
                        row = result.iloc[i + j]
                        col = cols[j]

                        with col:
                            img_url = row.get("image", "")
                            if isinstance(img_url, str) and img_url.startswith("http"):
                                st.image(img_url, use_container_width=True)
                            else:
                                st.image(DEFAULT_IMAGE_URL, use_container_width=True)

                            st.markdown(f"**{row['product_name']}**")
                            st.markdown(f"💰 **Giá:** {row['price']:,} đ")
                            st.markdown(f"📄 *{row['description'][:100]}...*")
                            st.markdown("---")
            