from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import joblib
from sklearn.decomposition import PCA
from skimage.feature import graycomatrix, graycoprops
import random


def process_image(image_path):
    image = Image.open(image_path)

    # Langkah 1: Potong gambar
    crop_coordinates = (1200, 1200, 2020, 1750)
    left, upper, right, lower = crop_coordinates
    cropped_image = image.crop((left, upper, right, lower))

    # Langkah 2: Konversi ke grayscale
    gray_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2GRAY)

    # Langkah 3: Thresholding
    thresholded_image = gray_image.copy()
    threshold_value = 120
    thresholded_image[gray_image > threshold_value] = 255

    #plt.imshow(thresholded_image, cmap='gray') 
    #plt.axis('off')
    #plt.show()

    return thresholded_image

def model_1(image):
    pca_model_1 = r'Code Philip MF/model/pca_model_1.pkl'
    classifier_model_1 = r'Code Philip MF/model/model_1_RF.h5'
    
    pca = joblib.load(pca_model_1)
    rf_classifier = joblib.load(classifier_model_1)
    
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    X = hist[20:100]
    X = X.reshape(1, -1)
    
    X_transformed = pca.transform(X)
    
    X_transformed_subset = X_transformed[:, 0:3]
    predictions = rf_classifier.predict(X_transformed_subset)

    if predictions == [0]:
        kesimpulan = 'No'
    else :
        kesimpulan = "Yes"

    return kesimpulan

def calculate_glcm_properties(image):
    distances = [1]
    
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm_properties = []
    glcm_energies = []
    glcm_homogeneities = []

    for angle in angles:
        glcm = graycomatrix(image, distances=distances, angles=[angle], symmetric=True, normed=True)
        
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        glcm_energies.append(energy)
        glcm_homogeneities.append(homogeneity)

    return glcm_energies, glcm_homogeneities


def model_2(image):
    pca_model_2 = r'Code Philip MF/model/pca_model_2.pkl'
    classifier_model_2 = r'Code Philip MF/model/model_2_RF.h5'
    
    pca = joblib.load(pca_model_2)
    rf_classifier = joblib.load(classifier_model_2)
    
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    X = hist[0:50]
    
    energy_values, homogeneity_values = calculate_glcm_properties(image)
    
    #print(X)
    #print(energy_values)
    #print(homogeneity_values)

    combined_data = np.hstack((energy_values, homogeneity_values, X))

    X = combined_data.reshape(1, -1)
    X_gabungan = pca.transform(X)        
    X_gabungan = X_gabungan[:, 0:4]
    
    predictions = rf_classifier.predict(X_gabungan)

    if predictions == [0]:
        kesimpulan = 'No'
    else :
        kesimpulan = "Yes"

    return kesimpulan

def model_3(image_asli):
    image = np.copy(image_asli)
    image[image == 255] = 0
    
    num_objects = 0
    
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corner_coordinates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x2, y2 = x + w, y + h
        
        contour_area = cv2.contourArea(contour)
        
        min_contour_area = 1
        if contour_area >= min_contour_area:
            corner_coordinates.append(((x, y), (x2, y2)))
            
    if corner_coordinates:
        image_with_boxes = cv2.cvtColor(image_asli, cv2.COLOR_GRAY2BGR)
        for i, ((x, y), (x2, y2)) in enumerate(corner_coordinates, start=1):
            cv2.rectangle(image_with_boxes, (x, y), (x2, y2), (0, 0, 255), 2)
        
        num_objects = len(corner_coordinates)

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title('Vision System Detection')
        plt.axis('off')
        plt.show()
    else:
        print("No objects were found in the image.")

    return num_objects

def cek_botol(image_path):
    start_time = time.time()
    result_image = process_image(image_path)
    result_1 = model_1(result_image)
    result_2 = model_2(result_image)
    model3 = model_3(result_image)

    if model3 == 12 :
        result_3 = 'No'
    else:
        result_3 = 'Yes'

    if result_1 == 'Yes' or result_2 == 'Yes' or result_3 == 'Yes':
        kesimpulan = "Botol Rejected"
    else:
        kesimpulan = "Botol Good"

    waktu = time.time() - start_time

    print("    Philip Bottle Vision System     ")
    print("------------------------------------")
    print("")
    print("Objek Terdeteksi : ", model3)
    print("")
    print("Cat Pudar     : ", result_1)
    print("Print Kurang  : ", result_2)
    print("Over Printing : ", result_3)
    print("")
    print("------------------------------------")
    print("Kesimpulan : ", kesimpulan)
    print("Deteksi Selesai Dalam", "{:.2f}".format(waktu), "detik")

image_path = r"Code Philip MF/Sebagian Dataset yang Digunakan/reject/REJECT_AVENT20230728113951796247.jpg"
image_path = r"Code Philip MF/Sebagian Dataset yang Digunakan/good/GOOD_AVENT20230728112550235490.jpg"

cek_botol(image_path)