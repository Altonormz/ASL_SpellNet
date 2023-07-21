import gradio as gr
import os

def video_identity(video):
    phrase = 'YONATAN'
    return video, phrase

# Set up the path to your local video file
video_path = os.path.join(os.path.abspath(''), "C:/Users/ariel/Desktop/Data Science/ITC/milestone3/deployment/fingerspelling_yonatan.mp4")

demo = gr.Interface(video_identity,
                    gr.Video(),[gr.Video(), "text"],
                    examples=[video_path],
                    cache_examples=True)

if __name__ == "__main__":
    demo.launch(share=True)
