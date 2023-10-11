import gradio as gr
import subprocess

def infer (prompt):
  command = [
        "python",
        "inference.py",
        f"--prompt={prompt}",
        "--output=output.gif" 
    ]

  try:
      subprocess.run(command, check=True)
  except subprocess.CalledProcessError as e:
      print(f"An error occurred: {e}")

  return "output.gif"

css="""
#col-container{
    margin: 0 auto;
    max-width: 720px;
    text-align: left;
}
"""

with gr.Blocks(css=css) as demo:
  with gr.Column(elem_id="col-container"):
    gr.HTML("""
<h2 style="text-align: center;">Hotshot-XL Text to GIF</h2>
<p style="text-align: center;">Hotshot-XL is an AI text-to-GIF model trained to work alongside Stable Diffusion XL</p>
        """)
    prompt = gr.Textbox(label="Prompt")
    submit_btn = gr.Button("Submit")
    gif_result = gr.Image(label="Gif result")
  submit_btn.click(fn=infer, inputs=[prompt], outputs=[gif_result])

demo.queue(max_size=12).launch()
    
