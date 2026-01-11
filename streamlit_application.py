import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pickle
import google.generativeai as genai

def load_model():
    try:
        model = joblib.load('breast_cancer_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        outlier_bounds = joblib.load('outlier_bounds.pkl')
        return model, scaler, feature_names,outlier_bounds
    except FileNotFoundError:
        st.error("Không tìm thấy file model. Vui lòng đảm bảo các file .pkl đã được tạo.")
        return None, None, None,None


def get_clean_data():
    try:
        data = pd.read_csv("breast-cancer.csv")
        data = data.drop('id', axis=1)
        data['diagnosis'] = np.where(data['diagnosis'] == 'M', 1, 0)
        return data
    except:
        return None


def add_sidebar():
    st.sidebar.header('Thông số của tế bào')
    
    # Lấy dữ liệu để xác định min/max 
    data = get_clean_data()
    
    slider_labels = [
        ("Bán kính (trung bình)", "radius_mean"),
        ("Độ nhám (trung bình)", "texture_mean"),
        ("Chu vi (trung bình)", "perimeter_mean"),
        ("Diện tích (trung bình)", "area_mean"),
        ("Độ mượt (trung bình)", "smoothness_mean"),
        ("Độ nén (trung bình)", "compactness_mean"),
        ("Độ lõm (trung bình)", "concavity_mean"),
        ("Điểm lõm (trung bình)", "concave points_mean"), #
        ("Độ đối xứng (trung bình)", "symmetry_mean"),
        ("Chiều fractal (trung bình)", "fractal_dimension_mean"),
        ("Bán kính (sai số)", "radius_se"),
        ("Độ nhám (sai số)", "texture_se"),
        ("Chu vi (sai số)", "perimeter_se"),
        ("Diện tích (sai số)", "area_se"), #
        ("Độ mượt (sai số)", "smoothness_se"),
        ("Độ nén (sai số)", "compactness_se"),
        ("Độ lõm (sai số)", "concavity_se"),
        ("Điểm lõm (sai số)", "concave points_se"),
        ("Độ đối xứng (sai số)", "symmetry_se"),
        ("Chiều fractal (sai số)", "fractal_dimension_se"),
        ("Bán kính (tệ nhất)", "radius_worst"), #
        ("Độ nhám (tệ nhất)", "texture_worst"), #
        ("Chu vi (tệ nhất)", "perimeter_worst"),
        ("Diện tích (tệ nhất)", "area_worst"), #
        ("Độ mượt (tệ nhất)", "smoothness_worst"), #
        ("Độ nén (tệ nhất)", "compactness_worst"),
        ("Độ lõm (tệ nhất)", "concavity_worst"),
        ("Điểm lõm (tệ nhất)", "concave points_worst"), #
        ("Độ đối xứng (tệ nhất)", "symmetry_worst"), #
        ("Chiều fractal (tệ nhất)", "fractal_dimension_worst")
    ]

    input_dict = {}
    for label, key in slider_labels:
       col_data = data[key]
       input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(col_data.min()),
            max_value=float(col_data.max()) ,
            value=float(col_data.mean()),
            format="%.4f")
    return input_dict


def get_scaled_values(input_dict, scaler):
    input_df = pd.DataFrame([input_dict])
    # Scale dữ liệu
    scaled_array = scaler.transform(input_df)
    
    return scaled_array


def add_predictions(input_data, model, scaler,outlier_bounds):
    input_df = pd.DataFrame([input_data])
    try:
        # Xử lý outliner
        clipped_features = []
        for col in input_df.columns:
            lower = outlier_bounds[col]['lower']
            upper = outlier_bounds[col]['upper']
            
            original_value = input_df[col].values[0]
            clipped_value = np.clip(original_value, lower, upper)
            
            input_df[col] = clipped_value

        # Chuẩn hóa dữ liệu
        scaled_data = scaler.transform(input_df)
        
        # Dự đoán
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)
        
        st.subheader("🔬 Kết quả dự đoán")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 0:
                st.success("✅ **LÀNH TÍNH (Benign)**")
                st.markdown(f"### Độ tin cậy: {probability[0][0]*100:.2f}%")
            else:
                st.error("⚠️ **ÁC TÍNH (Malignant)**")
                st.markdown(f"### Độ tin cậy: {probability[0][1]*100:.2f}%")
        
        with col2:
            # Biểu đồ xác suất
            prob_df = pd.DataFrame({
                'Loại': ['Lành tính', 'Ác tính'],
                'Xác suất (%)': [probability[0][0]*100, probability[0][1]*100]
            })
            st.bar_chart(prob_df.set_index('Loại'))
        
        # Hiển thị chi tiết xác suất
        st.write("---")
        st.write("**📊 Phân bố xác suất chi tiết:**")
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(
                label="🟢 Lành tính (Benign)", 
                value=f"{probability[0][0]*100:.2f}%"
            )
        with metric_col2:
            st.metric(
                label="🔴 Ác tính (Malignant)", 
                value=f"{probability[0][1]*100:.2f}%"
            )
        
        st.write("---")
        st.warning("""
        ⚠️ **LƯU Ý QUAN TRỌNG:**
        - Kết quả này chỉ mang tính chất tham khảo từ mô hình AI
        - Model có F1 ~96-98% trên tập test
        - Vui lòng tham khảo ý kiến bác sĩ chuyên khoa để có chẩn đoán chính xác
        - Không tự ý điều trị dựa trên kết quả này
        """)
        
        # Thông tin về model
        with st.expander("ℹ️ Thông tin về mô hình"):
            st.write("""
            **Mô hình:** Logistic Regression
            
            **Thông số:**
            - Solver: liblinear
            - Penalty: L2 (Ridge)
            - Regularization (C): 0.1
            - Class Weight: Balanced
            
            **Hiệu suất:**
            - Cross-Validation F1: ~96-98%
            - Test F1: ~96-98%
            
            **Dữ liệu huấn luyện:**
            - Dataset: Breast Cancer Wisconsin
            - Số mẫu: 569 bệnh nhân
            - Số đặc trưng: 30 đặc trưng từ hình ảnh nhân tế bào
            """)
        
    except Exception as e:
        st.error(f"❌ Có lỗi xảy ra khi dự đoán: {str(e)}")
        st.info("💡 Vui lòng kiểm tra lại dữ liệu đầu vào và các file model")


def main():
    st.set_page_config(
        page_title="Dự đoán Ung Thư Vú",
        layout="wide", 
        page_icon="🩺"
    )

    # Load model
    model, scaler, feature_names,outlier_bounds = load_model()

    # Header
    with st.container():
        st.title("🩺 Dự đoán Ung thư vú sử dụng mô hình Logistic Regression")
        st.write("""
        Ứng dụng sử dụng bộ dữ liệu **Breast Cancer Wisconsin** để hỗ trợ chẩn đoán sớm ung thư vú, 
        căn bệnh phổ biến hàng đầu ở nữ giới. Bằng cách phân tích các đặc trưng nhân tế bào qua 
        thuật toán **Logistic Regression**, hệ thống giúp chuyển hóa các chỉ số y khoa phức tạp 
        thành kết quả dự đoán khối u lành tính hoặc ác tính.
        """)
        st.info("💡 **Hướng dẫn:** Điều chỉnh các thanh trượt bên trái để nhập các chỉ số từ kết quả xét nghiệm, sau đó nhấn nút 'Dự đoán' bên dưới.")
    
    # Tạo sidebar
    sidebar_data = add_sidebar()
    
    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Dữ liệu đã nhập")
        st.write("Một số giá trị chính từ thanh trượt:")
        st.write("* Lưu ý điều chỉnh các biến: Những biến như diện tích, chu vi, bán kính, kết cấu có thể gây ảnh hưởng lớn hơn với mô hình !!!")
        
        # Hiển thị các giá trị quan trọng
        st.metric("Điểm lõm (trung bình)", f"{sidebar_data['concave points_mean']:.4f}")
        st.metric("Diện tích (sai số)", f"{sidebar_data['area_se']:.4f}")
        st.metric("Bán kính (tệ nhất)", f"{sidebar_data['radius_worst']:.4f}")
        st.metric("Độ nhám (tệ nhất)", f"{sidebar_data['texture_worst']:.4f}")
        st.metric("Diện tích (tệ nhất)", f"{sidebar_data['area_worst']:.4f}")
        st.metric("Độ mượt (tệ nhất)", f"{sidebar_data['smoothness_worst']:.4f}")
        st.metric("Điểm lõm (tệ nhất)", f"{sidebar_data['concave points_worst']:.4f}")
        st.metric("Độ đối xứng (tệ nhất)", f"{sidebar_data['symmetry_worst']:.4f}")
        
    with col2:
        st.subheader("📋 Thông tin")
        st.write(f" * Tổng số biến đã nhập: **{len(sidebar_data)}**")
        st.write(f" * Model đã load: **Logistic Regression**")
        st.write(f" * Scaler đã load: **StandardScaler**")
        st.write(f" * F1 của model: **~96-98%**")
    
    # Nút dự đoán
    st.write("---")
    if st.button("🔍 DỰ ĐOÁN KẾT QUẢ", type="primary", use_container_width=True):
        with st.spinner("⏳ Đang phân tích dữ liệu và dự đoán..."):
            add_predictions(sidebar_data, model, scaler, outlier_bounds)


if __name__ == "__main__":
    main()