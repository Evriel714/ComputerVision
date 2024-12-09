import gradio as gr
import pickle

def process_image():
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']

    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: '1', 25: '2', 26: '3', 27: '4', 28: '5', 29: '6'}


    
    return 


demo = gr.Interface(fn=process_image, inputs=gr.Image(type="filepath"), outputs="text")
demo.launch()   