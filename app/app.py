import gradio as gr
from fastai.vision.all import *

df = pd.read_csv('HAM10000_metadata.csv')

# Diccionario para mapear abreviaturas a nombres completos de lesiones
short_to_full_name_dict = {
    "akiec": "bowen's disease",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis-like lesions",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevi",
    "vasc": "vascular lesions",
}

# Reemplazar abreviaturas por nombres completos en la columna 'tipo_lesion'
df['lesion_type'] = df['dx'].map(short_to_full_name_dict)

# Crear el diccionario que relaciona cada imagen con su enfermedad
id_disease_dict = dict(zip(df['image_id'], df['lesion_type']))

# Definir una función para obtener las etiquetas (enfermedades) de las imágenes
def get_label(file_path):
    file_name = file_path.name
    image_id = file_name.split('.')[0]  # Obtener el ID de la imagen
    disease = id_disease_dict.get(image_id)  # Obtener la enfermedad asociada al ID
    return disease

learn = load_learner('convnext_small_in22k.pkl')

labels = learn.dls.vocab
def predict(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(labels,map(float,probs)))

image=gr.Image(type='pil')
label=gr.Label(num_top_classes=3)
title = "Skin Cancer Classifier"
description = "A skin cancer classifier trained on HAM100000 dataset."
examples = ['ex1_melanocitic_nevi.jpeg', 'ex2_melanoma.jpeg', 'ex3_basall_cell_carcinoma.jpeg']

demo = gr.Interface(fn=predict,inputs=image,outputs=label,title=title,description=description,examples=examples)

demo.launch(inline=False)
