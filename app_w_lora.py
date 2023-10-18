import os
import random
import subprocess
import tempfile
import gradio as gr
from huggingface_hub import snapshot_download, HfFileSystem, ModelCard

SECRET_TOKEN = os.getenv('SECRET_TOKEN', 'default_secret')

fs = HfFileSystem()

# execute a CLI command
def execute_command(command: str) -> None:
    subprocess.run(command, check=True)

def get_trigger_word(lora_id):
    # Get instance_prompt a.k.a trigger word
    card = ModelCard.load(lora_id)
    repo_data = card.data.to_dict()
    instance_prompt = repo_data.get("instance_prompt")

    if instance_prompt is not None:
        print(f"Trigger word: {instance_prompt}")
    else:
        instance_prompt = "no trigger word needed"
        print(f"Trigger word: no trigger word needed")
    return instance_prompt

def get_files(file_paths):
    last_files = {}  # Dictionary to store the last file for each path

    for file_path in file_paths:
        # Split the file path into directory and file components
        directory, file_name = file_path.rsplit('/', 1)
    
        # Update the last file for the current path
        last_files[directory] = file_name
    
    # Extract the last files from the dictionary
    result = list(last_files.values())

    return result


def load_lora_weights(lora_id):
    # List all ".safetensors" files in repo
    sfts_available_files = fs.glob(f"{lora_id}/*safetensors")
    sfts_available_files = get_files(sfts_available_files)

    if sfts_available_files == []:
        sfts_available_files = ["NO SAFETENSORS FILE"]

    print(f"Safetensors available: {sfts_available_files}")
    
    return sfts_available_files[0]


def infer(secret_token: str, prompt: str, negative_prompt: str, lora: str = None, size: str = '512x512', seed: int = -1, steps: int = 30, video_length: int = 8, video_duration: int = 1000):
    print(f"secret_token = {secret_token}\nprompt = {prompt}\nnegative_prompt = {negative_prompt}\nlora = {lora}\nsize = {size}\nseed = {seed}\nsteps = {steps}\n video_length = {video_length}\nvideo_duration = {video_duration}")
    if secret_token != SECRET_TOKEN:
        raise gr.Error(f'Invalid secret token. Please fork the original space if you want to use it for yourself.')
        
    width, height = map(int, size.split('x'))
    
    if seed < 0:
        seed = random.randint(0, 423538377342)
    
    if lora:  # only download if a link is provided
        print(f"lora model id: {lora}")
        lora_weights = load_lora_weights(lora)
        lora_path = lora
    else:
        lora_path = None

    # Use a temporary file instead of a static filename
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        output = tf.name

    command = [
      f"python", 
      f"inference.py",
      f"--prompt={prompt}",
      f"--negative_prompt={negative_prompt}",
      f"--output={output}",
      f"--width={width}",
      f"--height={height}",
      f"--seed={seed}",
      f"--steps={steps}",
      f"--video_length={video_length}",
      f"--video_duration={video_duration}",
    ]

    if lora:
        #command.append(f"--spatial_unet_base={spatial_unet_base}")
        command.append(f"--lora={lora_path}")
        command.append(f"--weight_name={lora_weights}")

    execute_command(command)

    return output
    
with gr.Blocks() as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""
            <div style="z-index: 100; position: fixed; top: 0px; right: 0px; left: 0px; bottom: 0px; width: 100%; height: 100%; background: white; display: flex; align-items: center; justify-content: center; color: black;">
              <div style="text-align: center; color: black;">
                <p style="color: black;">This space is a REST API to programmatically generate MP4s using a LoRA.</p>
                <p style="color: black;">Please see the <a href="https://hotshot.co" target="_blank">README.md</a> for more information.</p>
              </div>
        </div>""")
        secret_token = gr.Textbox(label="Secret token")
        prompt = gr.Textbox(label="Prompt")
        with gr.Accordion("Advanced Settings", open=False):
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                value="extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"
            )
            with gr.Row():
                lora = gr.Textbox(label="Public LoRA ID", value=None, placeholder="username/custom_lora_name")
                lora_trigger = gr.Textbox(label="Trigger word", interactive=False)
            with gr.Row():
                size = gr.Dropdown(
                    label="Size",
                    choices=[
                        '320x768',
                        '384x672',
                        '416x608',
                        '512x512',
                        '608x416',
                        '672x384',
                        '768x320',
                        '1024x1024',
                        '1024x512',
                        '1024x576'
                    ], value='512x512')
                
                seed = gr.Slider(
                    label="Seed",
                    info = "-1 denotes a random seed",
                    minimum=-1,
                    maximum=423538377342,
                    step=1,
                    value=-1
                )
                steps = gr.Slider(
                    label="Steps",
                    info = "Default is 30, but for high quality rendering values like 50 or 70 are good",
                    minimum=1,
                    maximum=423538377342,
                    step=1,
                    value=30
                )
                video_length = gr.Slider(
                    label="Video Length (FPS)",
                    info = "Good values are 1 (static image) and 8 (8 FPS) but bigger values aren't really supported",
                    minimum=1,
                    maximum=423538377342,
                    step=1,
                    value=8,
                )
                video_duration = gr.Slider(
                    label="Video Duration",
                    info = "it is 1000ms by default. You can try higher values but it may be buggy.",
                    minimum=1000,
                    maximum=423538377342,
                    step=1,
                    value=1000,
                )
        submit_btn = gr.Button("Submit")
        mp4_result = gr.Image(label="mp4")
    lora.blur(fn=get_trigger_word, inputs=[lora], outputs=[lora_trigger], queue=False)
    submit_btn.click(fn=infer, inputs=[secret_token, prompt, negative_prompt, lora, size, seed, steps, video_length, video_duration], outputs=[mp4_result])

demo.queue(max_size=12).launch()
