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


with gr.Blocks() as demo:
  with gr.Column(elem_id="col-container"):
    gr.HTML("""
        <div style="z-index: 100; position: fixed; top: 0px; right: 0px; left: 0px; bottom: 0px; width: 100%; height: 100%; background: white; display: flex; align-items: center; justify-content: center; color: black;">
          <div style="text-align: center;">
            <p>This space is a REST API to programmatically generate GIFs using a LoRA.</p>
            <p>Please see the <a href="https://hotshot.co" target="_blank">README.md</a> for more information.</p>
          </div>
    </div>""")
    prompt = gr.Textbox(label="Prompt")
    submit_btn = gr.Button("Submit")
    gif_result = gr.Image(label="Gif result")
  submit_btn.click(fn=infer, inputs=[prompt], outputs=[gif_result])

demo.queue(max_size=12).launch()
    
