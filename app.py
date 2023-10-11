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

demo = gr.Interface(fn=infer, inputs="textbox", outputs="file")
demo.launch()
    
