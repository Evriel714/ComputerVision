import gradio as gr

def process_image():
    return 


demo = gr.Interface(fn=process_image, inputs=gr.Image(type="filepath"), outputs="text")
demo.launch()   