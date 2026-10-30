# Intelligent Object Detection Using Deep Learning



## 1. Project Overview



This project focuses on developing an intelligent object detection system using deep learning and the YOLO (You Only Look Once) architecture.



The system detects and identifies multiple objects in images, videos, and live webcam streams. It also provides a web-based interface using Streamlit for interactive detection and visualization.



\*\*Project Title:\*\* Design and Development of Intelligent Object Detection using Deep Learning



\*\*Team Name:\*\* TrickedTriad



\*\*Team Members:\*\*

\- Sonali Chauhan

\- Nitya Jain

\- Shubhi Agrahari



\*\*Project Guide:\*\* Dr. Vivek Sharma



## 2. Objectives



\- Detect multiple objects in images and videos.

\- Perform real-time object detection using a webcam.

\- Evaluate and compare different YOLO models.

\- Analyze detection performance using standard evaluation metrics.

\- Develop an interactive web application for object detection.



## 3. Technologies Used



| Technology | Purpose |

|---|---|

| Python 3.11.9 | Core programming language |

| YOLO11 | Object detection |

| Ultralytics | Model implementation |

| PyTorch | Deep learning framework |

| OpenCV | Image and video processing |

| Streamlit | Web application interface |

| NumPy | Numerical operations |

| Matplotlib | Visualization |

| Jupyter Notebook | Experimentation and analysis |



## 4. Project Structure



```text

Intelligent\_Object\_Detection/

â”‚

â”œâ”€â”€ app/

â”‚   â””â”€â”€ main.py

â”‚

â”œâ”€â”€ evaluation/

â”‚   â”œâ”€â”€ evaluate\_yolo11n.py

â”‚   â”œâ”€â”€ evaluate\_yolo11s.py

â”‚   â”œâ”€â”€ compare\_yolo11n\_resolution.py

â”‚   â”œâ”€â”€ compare\_yolo11s\_confidence.py

â”‚   â”œâ”€â”€ compare\_sample\_predictions.py

â”‚   â”œâ”€â”€ convert\_coco\_to\_yolo.py

â”‚   â”œâ”€â”€ prepare\_subset.py

â”‚   â”œâ”€â”€ create\_dataset\_yaml.py

â”‚   â”œâ”€â”€ fix\_dataset\_yaml.py

â”‚   â”œâ”€â”€ yolo11n\_baseline.json

â”‚   â”œâ”€â”€ yolo11s\_results.json

â”‚   â”œâ”€â”€ yolo11n\_resolution\_comparison.json

â”‚   â””â”€â”€ yolo11s\_confidence\_comparison.json

â”‚

â”œâ”€â”€ input/

â”‚   â”œâ”€â”€ images/

â”‚   â””â”€â”€ videos/

â”‚

â”œâ”€â”€ models/

â”œâ”€â”€ notebooks/

â”œâ”€â”€ output/

â”œâ”€â”€ utils/

â”‚

â”œâ”€â”€ yolo11n.pt

â”œâ”€â”€ yolo11s.pt

â”œâ”€â”€ PROJECT\_PLAN.md

â”œâ”€â”€ .gitignore

â””â”€â”€ README.md

```



## 5. Models Evaluated



Two pretrained YOLO11 models were evaluated:



\- \*\*YOLO11n (Nano):\*\* Lightweight model designed for faster inference.

\- \*\*YOLO11s (Small):\*\* Larger model designed to provide improved detection performance.



Both models were evaluated using the same 500-image COCO validation subset, an image size of 640, batch size of 2, and CPU inference.



## 6. Evaluation Metrics



The following metrics were used:



\- \*\*Precision:\*\* Measures how many predicted detections are correct.

\- \*\*Recall:\*\* Measures how many actual objects are successfully detected.

\- \*\*F1-score:\*\* Harmonic mean of precision and recall.

\- \*\*mAP@50:\*\* Mean Average Precision at an IoU threshold of 0.50.

\- \*\*mAP@50â€“95:\*\* Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95.

\- \*\*Inference Time:\*\* Average time required to process one image.



## 7. Model Comparison Results



The following results were obtained on the 500-image evaluation subset.



| Metric | YOLO11n | YOLO11s |

|---|---:|---:|

| Precision | 66.07% | 72.02% |

| Recall | 51.07% | 59.45% |

| F1-score | 57.61% | 65.13% |

| mAP@50 | 57.51% | 66.03% |

| mAP@50â€“95 | 41.82% | 49.15% |

| Inference Time | 74.1 ms | 184.0 ms |



### Observations



\- YOLO11s achieved better precision, recall, F1-score, and mAP values.

\- YOLO11n demonstrated faster inference on the available CPU.

\- YOLO11s required approximately 2.48 times more inference time per image.



### Model Selection



YOLO11n is retained as the default application model because it provides faster inference and better responsiveness on CPU-based hardware.



YOLO11s is also included as an accuracy-focused alternative for future evaluation and deployment.



This selection considers the balance between detection quality and processing speed.



## 8. Additional Experiments



### Image Resolution Comparison



YOLO11n was evaluated at image sizes of 640 and 960.



The higher resolution produced only modest improvements in some metrics while increasing inference time considerably. Therefore, image size 640 was retained.



### Confidence Threshold Comparison



YOLO11s was evaluated using confidence thresholds of 0.15, 0.25, and 0.40.



The threshold of 0.25 produced the highest F1-score in this experiment, although the differences were small.



## 9. Dataset



The evaluation was performed using a randomly selected subset of 500 images from the COCO 2017 validation dataset.



\- Images: 500

\- Ground-truth annotations: 3,842

\- Object categories: 80

\- Non-empty YOLO label files: 494

\- Empty YOLO label files: 6



The dataset was converted into YOLO-compatible annotation format.



\*\*Note:\*\* This is a preliminary evaluation on a 500-image subset, not a complete evaluation on the full COCO validation dataset. The dataset files are not included in this repository.



## 10. Installation and Setup



### Prerequisites



\- Python 3.11.9

\- Git

\- Windows, Linux, or macOS



### Step 1: Clone the Repository



```bash

git clone https://github.com/sonali-chauhan8/Intelligent-Object-Detection\_5thsem.git

```



### Step 2: Open the Project Directory



```bash

cd Intelligent-Object-Detection\_5thsem

```



### Step 3: Create a Virtual Environment



```bash

python -m venv .venv

```



### Step 4: Activate the Environment



For Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



### Step 5: Install Dependencies



```bash

python -m pip install ultralytics streamlit opencv-python

```



## 11. Running the Application



Start the Streamlit application using:



```bash

python -m streamlit run app/main.py

```



The application provides the following modules:



1\. Dashboard

2\. Image Detection

3\. Video Detection

4\. Live Webcam Detection

5\. Analytics



The default application model is `yolo11n.pt`, located in the project root directory.



## 12. Future Scope



\- Fine-tuning the model on a domain-specific dataset.

\- Improving detection performance in crowded and complex environments.

\- Optimizing inference for CPU and GPU deployment.

\- Adding object tracking and advanced analytics.

\- Enhancing the user interface and reporting capabilities.

\- Exploring deployment on mobile and edge devices.



## 13. Conclusion



The project demonstrates an intelligent object detection system using YOLO11 and a Streamlit-based interface.



The evaluation highlights the trade-off between detection performance and inference speed. YOLO11s achieved stronger evaluation metrics, whereas YOLO11n provided faster processing on CPU hardware.



The project establishes a foundation for further improvements in object detection, tracking, and real-world deployment.



\---



\*\*Project Repository:\*\*  

https://github.com/sonali-chauhan8/Intelligent-Object-Detection\_5thsem


