# FGVC Aircraft Classification Project
The FGVC (Fine-Grained Visual Categorization) airplane project aimed to tackle the challenging task of classifying images of airplanes into distinct categories. Leveraging deep learning techniques, we explored various convolutional neural network (CNN) architectures and other models to achieve accurate and robust classification. The dataset, consisting of thousands of airplane images, posed a fine-grained categorization challenge, demanding models capable of identifying differences between aircraft types.

We started by experimenting with CNN models, progressing to more advanced architectures such as ResNet and InceptionV3. The continous process of fine-tuning these models grasped the capability to extract unique features to determine the difference between airplane classes. We encountered challenges related to model complexity and data augmentation

Some models demonstrated substantial accuracy improvements, others highlighted the importance of model complexity in handling fine-grained distinctions. The incorporation of transfer learning, ensemble methods, and adjustments in hyperparameters were pivotal in refining our models.

Despite facing challenges, the project served as a valuable exploration of deep learning methodologies in the context of fine-grained visual categorization. The significance of model interpretability, parameter tuning, and architectural choices became apparent. The project's scope encompassed not only achieving high accuracy but also understanding the limitations, providing a comprehensive view of the complexities inherent in the fine-grained categorization of airplane images.

# Vehicle Speed Detection using DeepStream and OpenCV

This project implements a vehicle speed detection system using NVIDIA's DeepStream SDK in combination with OpenCV for image processing. The goal is to track vehicles in a video stream, estimate their speed, and output the results in real-time. It leverages the power of GPU-accelerated deep learning and video analytics for efficient processing.

The pipeline processes a video input, detects vehicles, and tracks their movement across frames. By analyzing the vertical movement (y-coordinate) of the vehicles within the frame, the system estimates their speed. This is done by applying a perspective transformation to the detected objects, allowing for accurate speed calculations based on their motion relative to the camera's view.

Key technologies used in this project include:
DeepStream SDK: A framework by NVIDIA for building AI-powered video analytics applications. It enables efficient object detection and tracking using pre-trained deep learning models.
OpenCV: A powerful computer vision library used for perspective transformations and other image processing tasks.
GStreamer: A multimedia framework used to handle video input and output streams, enabling real-time video processing and manipulation.
CUDA: Provides GPU acceleration for faster computation, essential for handling large video streams and running deep learning models in real time.

The project is designed to be highly efficient, utilizing hardware acceleration and optimized software libraries to deliver real-time vehicle tracking and speed detection. It is ideal for applications in traffic monitoring, autonomous driving systems, and smart city infrastructure.

# Models training and testing (Data Analysis project)

The primary goal of this project is to develop a robust data analysis and machine learning pipeline to predict user clicks based on user and product features. The project aims to clean and preprocess the data, explore various machine learning models, and evaluate their performance to identify the best model for predicting user behavior. This involves handling missing data, merging datasets, and applying multiple machine learning algorithms to achieve accurate predictions.

Project Workflow

1. Data Reading and Cleaning: The project begins by reading three CSV files: `user_features.csv`, `product_features.csv`, and `click_history.csv`. These files contain essential information about users, products, and their interaction history. The data is then cleaned by filling missing values, dropping irrelevant records, and merging the datasets based on common keys.

2. Data Preprocessing: After cleaning, the data is preprocessed to ensure it is suitable for machine learning models. This includes handling missing values, normalizing data, and transforming categorical variables into numerical formats.

3. Model Training and Evaluation: The project explores several machine learning models, including Logistic Regression, Gaussian Naive Bayes, Decision Tree, Neural Network, Support Vector Machines (Linear, RBF, Poly), Random Forest Classifier, and AdaBoost Classifier. Each model is trained on the preprocessed data and evaluated using metrics such as accuracy, precision, F1 score, and recall.

4. Model Comparison: The performance of each model is compared to identify the best-performing model for predicting user clicks. The results are printed and analyzed to understand the strengths and weaknesses of each approach.

Technologies Used

- Python: The primary programming language used for data analysis and machine learning.
- Pandas: A powerful data manipulation library used for reading, cleaning, and preprocessing data.
- NumPy: A library for numerical computations, used for handling arrays and mathematical operations.
- Scikit-learn: A machine learning library that provides tools for model training, evaluation, and comparison.
- Jupyter Notebook: An interactive environment for writing and running Python code, used to document and execute the project workflow.

This project demonstrates a comprehensive approach to data analysis and machine learning, from data cleaning and preprocessing to model training and evaluation. By exploring multiple machine learning models, the project aims to identify the best approach for predicting user clicks, providing valuable insights into user behavior and product interactions.


