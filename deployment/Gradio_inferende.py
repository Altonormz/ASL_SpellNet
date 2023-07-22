import gradio as gr
import os


def video_identity(video):
    phrase = 'Rana Sanchez'
    video_path = "videoplayback_with_landmarks.mp4"
    return video_path, phrase


iface = gr.Interface(video_identity,
                    gr.Video(source="upload"),
                    [gr.Video(source="upload"), gr.Textbox()])

if __name__ == "__main__":
    iface.launch()
