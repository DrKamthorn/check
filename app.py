import streamlit as st
from PIL import Image, ImageOps
from img_classification import teachable_machine_classification

st.set_page_config(layout="wide")
st.title("AI/ML/DL หน่วยตรวจสุขภาพ ศูนย์การแพทย์กาญจนาภิเษก")
st.header("โปรแกรมคัดกรองภาพเอ็กซเรย์ปอด")
st.text("Upload ภาพ jpg ที่ต้องการ")

uploaded_file = st.file_uploader("Choose a photo ...", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'keras_Model.h5')
    if label == 0:
        st.write("ไม่พบสิ่งผิดปกติ")
        #st.write("สงสัยสิ่งผิดปกติ แนะนำสืบค้นเพิ่มเติม")
    else:
        #st.write("ไม่พบสิ่งผิดปกติ")
        st.write("สงสัยสิ่งผิดปกติ แนะนำสืบค้นเพิ่มเติม")