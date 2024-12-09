import gradio as gr

<<<<<<< Updated upstream
def process_image():
    return 


demo = gr.Interface(fn=process_image, inputs=gr.Image(type="filepath"), outputs="text")
=======
def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs=gr.Image(), outputs="text")
>>>>>>> Stashed changes
demo.launch()   