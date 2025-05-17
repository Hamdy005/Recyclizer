# ♻️ Recyclizer

An AI-powered image classification app that identifies and categorizes waste into four classes: **Recyclable, Non-Recyclable, Hazardous, and Organic**. Using transfer learning with a fine-tuned ResNet50 model, it achieves **97% training accuracy**, **95% validation accuracy**, and strong F1 scores.

---

## 📊 Features

- High-accuracy classification of waste images into four categories  
- Visualization of dataset distribution to analyze class balance  
- Training and validation accuracy curves to monitor model learning  
- Confusion matrix and classification report for detailed performance analysis  

---

## 🗂️ Waste Categories

- **Recyclable:** Plastic, Cardboard, Paper, Metal, Glass, Clothes, Shoes  
- **Non-Recyclable:** Everyday litter (e.g., chip bags, candy wrappers, used tissues), toothpaste tubes, disposable face masks  
- **Hazardous:** Batteries and other potentially harmful materials  
- **Organic:** Food waste and biological materials  

---

## 🚀 Deployment

- Interactive web demo built with **Streamlit**  
- Planned mobile deployment  

---

## 🛠️ Setup & Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/recyclizer.git
   cd recyclizer

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
  
3. Run the Streamlit app locally:

   ```bash
   streamlit run app.py
   
Or try the live demo here:
https://recyclizer.streamlit.app/
  
