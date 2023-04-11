import streamlit as st
import cv2
from harris import* 
from matching import*
from SIFT import*
st.set_page_config(page_title=" Image Processing", page_icon="📸", layout="wide",initial_sidebar_state="collapsed")

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_st_style, unsafe_allow_html=True)

with open("style.css") as source_des:
    st.markdown(f"""<style>{source_des.read()}</style>""", unsafe_allow_html=True)

side = st.sidebar
uploaded_img =side.file_uploader("Upload Image",type={"png", "jpg", "jfif" , "jpeg"})
threshold = side.number_input('Harris Threshold',min_value=10,max_value=3000000, value=10000,step=10)
k = side.number_input('Harris k value',min_value=.00001,max_value=1.0, value=.04,step=.00001)
side.text('SIFT parameters')
window_size = side.number_input('Harris window size',min_value=1,max_value=15, value=5,step=1)
Sigma=side.number_input('Sigma of Gaussian filter',value=1.6)
num_of_octaves=side.number_input('Number of octaves',value=4)

tab1, tab2  = st.tabs(["Harris & SIFT", "Matching"])
with tab1:
     
    col1,col2 = st.columns(2)
    select=col2.selectbox("Select",('','Harris','SIFT'))
    if uploaded_img is not None:
        file_path = 'Images/'  +str(uploaded_img.name)
        input_img = cv2.imread(file_path)
        gray_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        sized_img = cv2.resize(input_img,(400,400))
        col1.image(sized_img)
        if select=="Harris":
            corner_list, corner_img = find_harris_corners(gray_image, k,window_size,threshold)
            output_image = cv2.resize(corner_img,(400,400))
            col2.image(output_image)
        if select=="SIFT":
            KeyPoints,descriptors= computeKeypointsAndDescriptors(gray_image,num_of_octaves,Sigma)  
            # draw keypoints in image
            output_image= cv2.drawKeypoints(gray_image,KeyPoints, None, flags=0)  
            output_image = cv2.resize(output_image,(400,400))
            col2.image(output_image)    

with tab2:
    uploadimg1, uploadimg2, col3 = st.columns(3)
    select = col3.selectbox("Select",('SSD','NCC'))
    img1 = uploadimg1.file_uploader("upload Image", type = {"png","jpg","jfif", "jpeg"}, key="tab12")
    if img1 is not None:
        file_path = 'Images/'  +str(img1.name)
        input_img1 = cv2.imread(file_path)
        gray_image1 = cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
        sized_img1 = cv2.resize(input_img1,(400,400))
        # uploadimg1.image(sized_img1)
    img2 = uploadimg2.file_uploader("upload Image", type = {"png","jpg","jfif", "jpeg"}, key="tab22")
    if img2 is not None:
        file_path = 'Images/'  +str(img2.name)
        input_img2 = cv2.imread(file_path)
        gray_image2 = cv2.cvtColor(input_img2, cv2.COLOR_BGR2GRAY)
        sized_img2 = cv2.resize(input_img2,(400,400))
        # uploadimg2.image(sized_img2) 
        
        if select == "SSD" :
            montage = match_SSD (gray_image1,gray_image2)
            col3.image(montage)


           



