import gradio as gr
from inference import process_image
import numpy as np
import cv2

def gradio_infer(img):
    # img приходит как numpy array H×W×3
    vis, seed_cnt, imp_cnt, sa, ia = process_image(img)
    caption = (
        f"Семян: {seed_cnt}, Примесей: {imp_cnt}\n"
        f"Площадь семян: {sa}, Площадь примесей: {ia}"
    )
    # вернуть изображение в RGB
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption

demo = gr.Interface(
    fn=gradio_infer,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="numpy"), gr.Textbox()],
    title="Seed & Impurity Detector",
    description="Загрузи фото зерна — получишь разметку и статистику"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
